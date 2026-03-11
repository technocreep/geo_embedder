# ─────────────────────────────────────────────────────────────────────────────
# Geo Embedder — ML Pipeline Container
# Base: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
#
# Что добавляем поверх базового образа:
#   - системные либы для PDF (poppler) и DOCX (libxml2)
#   - Python-зависимости пайплайна
#   - non-root пользователь для безопасности
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# ── Метаданные ────────────────────────────────────────────────────────────────
LABEL maintainer="geo-embedder"
LABEL description="Fine-tuning & serving BAAI/bge-m3 for geology domain"
LABEL cuda="12.1"
LABEL pytorch="2.4.1"

# ── Системные переменные ──────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Пути внутри контейнера
ENV APP_DIR=/app
ENV DATA_DIR=/data
ENV MODELS_DIR=/models
ENV OUTPUT_DIR=/output
ENV HF_HOME=/models/huggingface

# ── Системные зависимости ─────────────────────────────────────────────────────
# - poppler-utils     : извлечение текста из PDF (используется pypdf)
# - libxml2-dev       : нужна python-docx
# - libxslt1-dev      : нужна lxml (зависимость docx)
# - git               : для pip install из git-репозиториев
# - curl              : отладка и healthcheck
# - build-essential   : компиляция C-расширений (rank-bm25 и др.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libxml2-dev \
        libxslt1-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        curl \
        wget \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root пользователь ────────────────────────────────────────────────────
RUN useradd -m -u 1000 -s /bin/bash geouser
RUN mkdir -p $APP_DIR $DATA_DIR $MODELS_DIR $OUTPUT_DIR \
    && chown -R geouser:geouser $APP_DIR $DATA_DIR $MODELS_DIR $OUTPUT_DIR

# ── Python-зависимости ────────────────────────────────────────────────────────
# Копируем отдельно от кода — чтобы Docker кэшировал слой при изменении кода
COPY --chown=geouser:geouser requirements.txt $APP_DIR/requirements.txt

# PyTorch уже в образе — устанавливаем только остальное
# torch и torchvision пропускаем через --constraint чтобы не переустанавливать
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        # ── данные ──────────────────────────────────
        pypdf>=4.0.0 \
        python-docx>=1.1.0 \
        langchain-text-splitters>=0.2.0 \
        rank-bm25>=0.2.2 \
        tqdm>=4.66.0 \
        # ── обучение ────────────────────────────────
        sentence-transformers>=2.7.0 \
        transformers>=4.40.0 \
        datasets>=2.18.0 \
        accelerate>=0.28.0 \
        # ── оценка ──────────────────────────────────
        scikit-learn>=1.4.0 \
        numpy>=1.26.0 \
        # ── сервинг ─────────────────────────────────
        fastapi>=0.111.0 \
        "uvicorn[standard]>=0.29.0" \
        pydantic>=2.6.0 \
        # ── LLM API ─────────────────────────────────
        openai>=1.30.0

# ── Копируем код проекта ──────────────────────────────────────────────────────
COPY --chown=geouser:geouser . $APP_DIR/

# ── Рабочая директория и пользователь ────────────────────────────────────────
WORKDIR $APP_DIR
USER geouser

# ── Healthcheck для serving-контейнера ───────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# ── По умолчанию — интерактивная оболочка (переопределяется в compose) ────────
CMD ["/bin/bash"]
