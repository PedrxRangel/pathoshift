"""PathoShift: classificação de patches H&E com robustez a mudança de domínio.

Uso (na ordem do pipeline):
    python pathoshift.py describe                      # config, versões e dispositivo
    python pathoshift.py prepare                       # baixa os dados, gera manifest, splits e folds
    python pathoshift.py features                      # features clássicas -> data/features.parquet
    python pathoshift.py train --model logreg          # validação cruzada (4 folds por paciente)
    python pathoshift.py train --model cnn
    python pathoshift.py train --model logreg --final  # modelo final no dev inteiro (não olha o teste)
    python pathoshift.py train --model cnn --final
    python pathoshift.py predict                       # outputs/predictions.jsonl (CV)
    python pathoshift.py predict --teste-externo       # + centro 2, UMA vez, com a config congelada
    python pathoshift.py evaluate                      # outputs/metrics.json + outputs/figuras/

Tempos, config, versões e GPU de cada etapa ficam em outputs/run_info.json.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd
import yaml

from src import utils


def cmd_prepare(cfg, args):
    """Baixa o dataset, gera data/manifest.jsonl com split e fold e imprime os resumos. Apoio."""
    from src.data import load_hf_dataset, load_metadata
    from src.manifest import build_manifest, save_manifest, summarize
    from src.splits import make_splits

    sp = cfg["splits"]
    with utils.timer("prepare", cfg):
        ds = load_hf_dataset(cfg)
        manifest = build_manifest(load_metadata(ds))
        manifest["split"], manifest["fold"] = make_splits(
            manifest, sp["test_centre"], sp["n_folds"], cfg["seed"], sp["min_tumor_patches_per_fold"]
        )
        out = utils.resolve_path(cfg["paths"]["manifest"])
        save_manifest(manifest, out)
    print(f"Dataset: {len(ds)} patches | manifest salvo em {out}")

    by_centre = summarize(manifest, "centre_id")
    by_fold = summarize(manifest, "fold").rename(index={-1: f"test_ext (c{sp['test_centre']})"})
    by_centre.to_csv(out.parent / "resumo_por_centro.csv")
    by_fold.to_csv(out.parent / "resumo_por_fold.csv")
    print("\n=== Resumo por centro ===")
    print(by_centre.to_string())
    print("\n=== Resumo por fold (dev) e teste externo ===")
    print(by_fold.to_string())
    return 0


def cmd_features(cfg, args):
    """Calcula as features clássicas e salva em parquet (--limit grava uma AMOSTRA em outro arquivo). Apoio."""
    from src.data import load_hf_dataset
    from src.features import compute_features, save_features
    from src.manifest import load_manifest

    man = load_manifest(utils.resolve_path(cfg["paths"]["manifest"]))
    out = utils.resolve_path(cfg["paths"]["features"])
    if args.limit is not None:
        # amostra para teste rápido: nunca sobrescreve o arquivo completo usado no treino
        ids = man["id"].sample(args.limit, random_state=cfg["seed"])
        out = out.with_name("features_amostra.parquet")
    else:
        ids = man["id"]
    ds = load_hf_dataset(cfg)
    with utils.timer("features" if args.limit is None else "features_amostra", cfg):
        feats = compute_features(ds, ids, cfg["features"])
        save_features(feats, out)
    print(f"{feats.shape[1] - 1} features x {len(feats)} patches salvas em {out}")
    return 0


def cmd_train(cfg, args):
    """Validação cruzada (padrão) ou modelo final no dev inteiro (--final). Apoio."""
    from src.train import run_cv, run_final

    step = f"train_{args.model}_{'final' if args.final else 'cv'}"
    with utils.timer(step, cfg):
        if args.final:
            run_final(cfg, args.model, epochs=args.epochs)
        else:
            run_cv(cfg, args.model, epochs=args.epochs)
    return 0


def cmd_predict(cfg, args):
    """Gera outputs/predictions.jsonl (CV dos dois modelos e, com --teste-externo, o centro 2). Apoio."""
    from src.predict import MODELS, build_predictions, predict_test, save_predictions

    info = utils.load_run_info(cfg)
    with utils.timer("predict", cfg):
        if args.teste_externo:
            if "teste_externo" in info and not args.forcar:
                print("O teste externo JÁ foi executado em "
                      f"{info['teste_externo']['executado_em']}. Ele deve ser usado uma vez só.\n"
                      "Se for mesmo necessário refazer, use --forcar (e registre o motivo no relatório).")
                return 1
            for name in MODELS:
                print(f"Teste externo: {name}")
                predict_test(cfg, name)
        preds = build_predictions(cfg)
        out = utils.resolve_path(cfg["paths"]["outputs_dir"]) / "predictions.jsonl"
        save_predictions(preds, out)

    if args.teste_externo:
        info = utils.load_run_info(cfg)
        config_file = utils.resolve_path(args.config)
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=utils.ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=utils.ROOT)
        info["teste_externo"] = {
            "executado_em": datetime.now().isoformat(timespec="seconds"),
            "centro": cfg["splits"]["test_centre"],
            "modelos": MODELS,
            "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
            "git_head": git.stdout.strip() or None,
            "codigo_com_alteracoes_nao_commitadas": bool(dirty.stdout.strip()),
            "declaracao": (f"Configuração congelada. A partir daqui, nenhuma escolha de hiperparâmetro, "
                           f"limiar ou pré-processamento usará o centro {cfg['splits']['test_centre']}."),
            "refeito_com_forcar": bool(args.forcar),
        }
        utils.save_run_info(cfg, info)
    print(f"{len(preds)} predições salvas em {out}")
    print(preds.groupby(["modelo", "origem"]).size().to_string())
    return 0


def _print_report(metrics):
    """Tabela resumida no terminal: CV (média ± desvio), por paciente e teste externo. Apoio."""
    rows = []
    for name, rep in metrics["modelos"].items():
        cvp, cvq = rep["cv"]["por_patch"], rep["cv"]["por_paciente"]
        row = {"modelo": name}
        for k in ["balanced_accuracy", "f1_macro", "auc", "recall_tumor", "recall_normal"]:
            row[f"CV {k}"] = f"{cvp[k]['media']:.3f} ± {cvp[k]['desvio']:.3f}"
        row["CV acc_bal pacientes"] = (f"{cvq['acuracia_balanceada_pacientes']['media']:.3f} ± "
                                       f"{cvq['acuracia_balanceada_pacientes']['desvio']:.3f}")
        te = rep.get("teste_externo")
        if te:
            for k in ["balanced_accuracy", "f1_macro", "auc", "recall_tumor", "recall_normal"]:
                row[f"C2 {k}"] = f"{te['por_patch'][k]:.3f}"
            row["C2 acc_bal pacientes"] = f"{te['por_paciente']['acuracia_balanceada_pacientes']:.3f}"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("modelo").T.to_string())


def cmd_evaluate(cfg, args):
    """Lê predictions.jsonl, calcula as métricas (patch, paciente, centro) e salva metrics.json e as figuras. Apoio."""
    import matplotlib
    matplotlib.use("Agg")
    from src import viz
    from src.data import load_hf_dataset
    from src.evaluate import build_metrics
    from src.manifest import load_manifest
    from src.predict import load_predictions

    out_dir = utils.resolve_path(cfg["paths"]["outputs_dir"])
    with utils.timer("evaluate", cfg):
        man = load_manifest(utils.resolve_path(cfg["paths"]["manifest"]))
        preds = load_predictions(out_dir / "predictions.jsonl")
        metrics = build_metrics(preds, man)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        fig_dir = out_dir / "figuras"
        fig_dir.mkdir(parents=True, exist_ok=True)
        ds = load_hf_dataset(cfg)
        cv = preds[preds["origem"] == "cv"].merge(man, on="id").rename(columns={"prob_tumor": "prob"})
        for name, rep in metrics["modelos"].items():
            viz.confusion_figure(rep, name).savefig(fig_dir / f"matriz_confusao_{name}.png", dpi=110)
            viz.errors_figure(ds, cv[cv["modelo"] == name], name).savefig(fig_dir / f"erros_{name}.png", dpi=110)
        hist_path = out_dir / "cnn" / "cv_history.csv"
        if hist_path.exists():
            viz.cnn_curves_figure(pd.read_csv(hist_path)).savefig(fig_dir / "curvas_cnn.png", dpi=110)
        viz.comparison_figure(metrics).savefig(fig_dir / "comparacao_modelos.png", dpi=110, bbox_inches="tight")

        # cópia das tabelas dos dados: assim o notebook de resultados só precisa de outputs/
        tab_dir = out_dir / "tabelas"
        tab_dir.mkdir(exist_ok=True)
        data_dir = utils.resolve_path(cfg["paths"]["data_dir"])
        for name in ["resumo_por_centro.csv", "resumo_por_fold.csv"]:
            if (data_dir / name).exists():
                shutil.copy(data_dir / name, tab_dir / name)
    _print_report(metrics)
    print(f"\nmetrics.json e figuras salvos em {out_dir}")
    return 0


def cmd_describe(cfg, args):
    """Mostra a configuração, as versões das bibliotecas e o dispositivo. Apoio."""
    print("=== Configuração ===")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("=== Versões ===")
    print(f"  sistema: {platform.platform()}")
    for pkg, ver in utils.package_versions().items():
        print(f"  {pkg:24s} {ver}")
    print("\n=== Dispositivo ===")
    for k, v in utils.gpu_info().items():
        print(f"  {k}: {v}")
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
    p.add_argument("--limit", type=int, default=None,
                   help="só N patches aleatórios, gravados em data/features_amostra.parquet (teste rápido)")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="validação cruzada (ou modelo final com --final)")
    p.add_argument("--model", choices=["logreg", "cnn"], required=True)
    p.add_argument("--epochs", type=int, default=None, help="máximo de épocas da CNN (padrão: config)")
    p.add_argument("--final", action="store_true", help="treina no dev inteiro e salva o modelo final")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="gera outputs/predictions.jsonl")
    p.add_argument("--teste-externo", action="store_true",
                   help="aplica os modelos finais ao centro de teste (UMA vez, com a config congelada)")
    p.add_argument("--forcar", action="store_true", help="refaz o teste externo mesmo já tendo rodado")
    p.set_defaults(func=cmd_predict)

    sub.add_parser("evaluate", help="gera outputs/metrics.json e outputs/figuras/").set_defaults(func=cmd_evaluate)
    sub.add_parser("describe", help="mostra config, versões e dispositivo").set_defaults(func=cmd_describe)

    args = parser.parse_args()
    cfg = utils.load_config(args.config)
    utils.set_seed(cfg["seed"])
    return args.func(cfg, args)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    sys.exit(main())
