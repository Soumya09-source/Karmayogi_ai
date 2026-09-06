from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.embed_and_load import embed_texts


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache()
def get_embedding_model():
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def embed_texts_for_app(texts: list[str]):
    model = get_embedding_model()
    return embed_texts(model, texts)