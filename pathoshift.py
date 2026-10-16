"""PathoShift: classificação de patches H&E com robustez a mudança de domínio.

Uso:
    python pathoshift.py describe                     # config, versões e dispositivo
    python pathoshift.py prepare                      # baixa os dados, gera manifest, splits e folds
    python pathoshift.py features                     # calcula as features clássicas (parquet)
    python pathoshift.py train --model logreg         # validação cruzada do baseline clássico
    python pathoshift.py train --model cnn --epochs 3 # validação cruzada da CNN
    python pathoshift.py train --model cnn --final    # treino no dev + teste ÚNICO no centro externo
    python pathoshift.py evaluate --model cnn         # métricas a partir das predições salvas
"""

import argparse
import platform
import sys

import pandas as pd
import yaml

from src import utils


def cmd_prepare(cfg, args):
    """Baixa o dataset, gera data/manifest.jsonl com split e fold e imprime os resumos. Apoio."""
    from src.data import load_hf_dataset, load_metadata
    from src.manifest import build_manifest, save_manifest, summarize
    from src.splits import make_splits

    sp = cfg["splits"]
    with utils.timer("download/abertura do dataset"):
        ds = load_hf_dataset(cfg)
    print(f"Dataset: {len(ds)} patches | colunas: {ds.column_names}")

    manifest = build_manifest(load_metadata(ds))
    manifest["split"], manifest["fold"] = make_splits(
        manifest, sp["test_centre"], sp["n_folds"], cfg["seed"], sp["min_tumor_patches_per_fold"]
    )
    out = utils.resolve_path(cfg["paths"]["manifest"])
    save_manifest(manifest, out)
    print(f"Manifest salvo em {out}")

    by_centre = summarize(manifest, "centre_id")
    by_fold = summarize(manifest, "fold").rename(index={-1: f"test_ext (c{sp['test_centre']})"})
    by_centre.to_csv(out.parent / "resumo_por_centro.csv")
    by_fold.to_csv(out.parent / "resumo_por_fold.csv")
    print("\n=== Resumo por centro ===")
    print(by_centre.to_string())
    print(f"\n=== Resumo por fold (dev) e teste externo ===")
    print(by_fold.to_string())
    return 0


def cmd_features(cfg, args):
    """Calcula as features clássicas de todos os patches e salva em parquet. Apoio."""
    from src.data import load_hf_dataset
    from src.features import compute_features, save_features
    from src.manifest import load_manifest

    man = load_manifest(utils.resolve_path(cfg["paths"]["manifest"]))
    ids = man["id"] if args.limit is None else man["id"].sample(args.limit, random_state=cfg["seed"])
    ds = load_hf_dataset(cfg)
    with utils.timer(f"features de {len(ids)} patches"):
        feats = compute_features(ds, ids, cfg["features"])
    out = utils.resolve_path(cfg["paths"]["features"])
    save_features(feats, out)
    print(f"{feats.shape[1] - 1} features x {len(feats)} patches salvas em {out}")
    print(feats.describe().T.round(3).to_string())
    return 0


def cmd_train(cfg, args):
    """Validação cruzada (padrão) ou treino final + teste externo (--final). Apoio."""
    from src.train import run_cv, run_final

    with utils.timer(f"train {args.model}{' final' if args.final else ''}"):
        if args.final:
            run_final(cfg, args.model)
        else:
            run_cv(cfg, args.model, epochs=args.epochs)
    return 0


def cmd_evaluate(cfg, args):
    """Recalcula as métricas a partir das predições salvas em outputs/<modelo>/. Apoio."""
    from src.evaluate import compute_metrics, fold_table, mean_std_text, print_metrics

    out_dir = utils.resolve_path(cfg["paths"]["outputs_dir"]) / args.model
    if args.final:
        preds = pd.read_csv(out_dir / "final_test_preds.csv")
        print_metrics(compute_metrics(preds["label"], preds["prob"]), f"{args.model} | teste externo")
        return 0
    oof = pd.read_csv(out_dir / "cv_oof.csv")
    table = fold_table([compute_metrics(g["label"], g["prob"]) for _, g in oof.groupby("fold")])
    print(table.round(3).to_string())
    print("\nmédia ± desvio:\n" + mean_std_text(table).to_string())
    print("\nPor centro (predições de validação de todos os folds juntas):")
    for c, g in oof.groupby("centre_id"):
        m = compute_metrics(g["label"], g["prob"])
        texto = f"  centro {c}: n={len(g):5d}  tumor={g['label'].sum():4d}  "
        if g["label"].nunique() == 2:
            texto += f"acc_bal={m['balanced_accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}  auc={m['auc']:.3f}"
        else:  # centro só com normais: a única métrica com sentido é a especificidade
            texto += f"só normais -> especificidade={m['recall_normal']:.3f}"
        print(texto)
    return 0


def cmd_predict(cfg, args):
    """Ainda não implementado: as predições já são salvas pelo `train`. Apoio."""
    print("predict: ainda não implementado. As predições ficam em outputs/<modelo>/ após o `train`.")
    return 1


def cmd_describe(cfg, args):
    """Mostra a configuração, as versões das bibliotecas e o dispositivo. Apoio."""
    print("=== Configuração ===")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    print("=== Versões ===")
    print(f"  sistema: {platform.platform()}")
    for pkg, ver in utils.package_versions().items():
        print(f"  {pkg:24s} {ver}")

    print("\n=== Dispositivo ===")
    import torch
    device = utils.get_device()
    print(f"  dispositivo: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} | {props.total_memory / 1024**3:.1f} GB | CUDA {torch.version.cuda}")
    else:
        print("  Nenhuma GPU CUDA encontrada: o treino vai rodar na CPU (lento).")

    freeze = utils.save_pip_freeze(utils.resolve_path(cfg["paths"]["outputs_dir"]) / "versions.txt")
    print(f"\nLista completa de pacotes salva em {freeze}")
    return 0


def main():
    """Lê os argumentos da linha de comando e chama o comando escolhido. Apoio."""
    parser = argparse.ArgumentParser(description="PathoShift Challenge (Processamento de Imagens)")
    parser.add_argument("--config", default="configs/default.yaml", help="arquivo YAML de configuração")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="baixa o dataset e gera manifest + folds").set_defaults(func=cmd_prepare)

    p = sub.add_parser("features", help="calcula as features clássicas")
    p.add_argument("--limit", type=int, default=None, help="só N patches aleatórios (teste rápido)")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="treina com validação cruzada (ou --final)")
    p.add_argument("--model", choices=["logreg", "cnn"], required=True)
    p.add_argument("--epochs", type=int, default=None, help="máximo de épocas da CNN (padrão: config)")
    p.add_argument("--final", action="store_true", help="treina no dev e testa no centro externo (só no fim!)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="métricas a partir das predições salvas")
    p.add_argument("--model", choices=["logreg", "cnn"], required=True)
    p.add_argument("--final", action="store_true", help="métricas do teste externo")
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("predict", help="(não implementado)").set_defaults(func=cmd_predict)
    sub.add_parser("describe", help="mostra config, versões e dispositivo").set_defaults(func=cmd_describe)

    args = parser.parse_args()
    cfg = utils.load_config(args.config)
    utils.set_seed(cfg["seed"])
    return args.func(cfg, args)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    sys.exit(main())
