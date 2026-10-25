FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir /data && chown app /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[llm,whisper]"

USER app
ENV KGV_DATA_DIR=/data
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["kgvoicebot", "serve", "--host", "0.0.0.0", "--port", "8000"]
