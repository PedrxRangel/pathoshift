# Esqueleto de Dockerfile (ainda não usado no fluxo do grupo).
# Build:  docker build -t pathoshift .
# Rodar:  docker run --rm pathoshift python pathoshift.py describe
FROM --platform=linux/amd64 python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências primeiro (aproveita o cache do Docker quando só o código muda)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "pathoshift.py", "describe"]
