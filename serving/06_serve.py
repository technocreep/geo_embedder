"""
06_serve.py
===========
FastAPI-сервер для дообученного эмбеддера.
Реализует OpenAI-совместимый эндпоинт POST /v1/embeddings.
Именно этот формат ожидает Open WebUI при настройке кастомного эмбеддера.

Запуск:
    pip install fastapi uvicorn sentence-transformers
    python 06_serve.py --model ./finetuned-bge-m3-geo --port 8080

Проверка:
    curl -X POST http://localhost:8080/v1/embeddings \
        -H "Content-Type: application/json" \
        -d '{"input": ["геологический разрез юрских отложений"], "model": "geo-embedder"}'

Настройка в Open WebUI:
    Admin Panel → Settings → RAG
    Embedding Model Engine  : OpenAI
    Embedding Base URL      : http://localhost:8080
    Embedding Model         : geo-embedder
    API Key                 : dummy (любая непустая строка)
"""

import argparse
import logging
import time
from typing import Union

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Схема запроса/ответа (OpenAI-compatible) ─────────────────────────────────

class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = "geo-embedder"
    encoding_format: str = "float"


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: dict


# ─── Глобальный объект модели ─────────────────────────────────────────────────

_model: SentenceTransformer = None
_model_name: str = ""


def get_model() -> SentenceTransformer:
    return _model


# ─── FastAPI приложение ───────────────────────────────────────────────────────

app = FastAPI(
    title="Geo Embedder API",
    description="Доменный эмбеддер для геологии и геологоразведки (bge-m3 fine-tuned)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info(f"Модель загружена: {_model_name}")
    logger.info(f"Устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Размерность эмбеддингов: {_model.get_sentence_embedding_dimension()}")


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": _model_name,
        "embedding_dim": _model.get_sentence_embedding_dimension(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/v1/models")
def list_models():
    """Open WebUI может запрашивать список моделей."""
    return {
        "object": "list",
        "data": [{
            "id": "geo-embedder",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    # Нормализация входа: str или list[str]
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    if not texts:
        raise HTTPException(status_code=400, detail="Пустой список текстов")

    if len(texts) > 512:
        raise HTTPException(status_code=400, detail="Максимум 512 текстов за запрос")

    try:
        t0 = time.time()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        elapsed = time.time() - t0
        logger.info(f"Закодировано {len(texts)} текстов за {elapsed:.2f}с")

        data = [
            EmbeddingObject(
                embedding=emb.tolist(),
                index=i,
            )
            for i, emb in enumerate(embeddings)
        ]

        # Примерный подсчёт токенов (символов / 4)
        total_tokens = sum(len(t) // 4 for t in texts)

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        )

    except Exception as e:
        logger.error(f"Ошибка кодирования: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "model": _model_name}


# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    global _model, _model_name
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./finetuned-bge-m3-geo",
                        help="Путь к дообученной модели или HuggingFace ID")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    _model_name = args.model
    logger.info(f"Загружаем модель: {_model_name}")
    _model = SentenceTransformer(_model_name)

    # Прогрев
    logger.info("Прогрев модели...")
    _ = _model.encode(["тест"], normalize_embeddings=True)
    logger.info(f"Сервер запускается на http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
