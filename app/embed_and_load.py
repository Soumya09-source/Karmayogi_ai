"""
embed_and_load.py

Generates sentence-transformers embeddings for:
  - concept_taxonomy_embedready.csv   (text column: embedding_text)
  - courses_master_embedready.csv     (text column: embedding_text)
  - document_chunks_embedready.csv    (text column: chunk_text)

and loads each into its matching pgvector-backed Postgres table
(schema.sql must be run once beforehand).

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # fill in your real DB credentials
    python embed_and_load.py --data-dir /path/to/csvs
"""

import argparse
import os
import sys

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# all-MiniLM-L6-v2: 384-dim, fast, good default for semantic similarity.
# Swap this for a stronger model (e.g. "all-mpnet-base-v2", 768-dim) if
# retrieval quality matters more than speed — just update VECTOR(384)
# to VECTOR(768) in schema.sql to match.
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def get_connection():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def embed_texts(model, texts):
    """Encode a list of strings in batches, return list of np.ndarray."""
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalized vectors -> cosine similarity
                                     # via inner product is equivalent and faster
    )
    return embeddings


def load_concept_taxonomy(conn, model, csv_path):
    df = pd.read_csv(csv_path)
    df["embedding_text"] = df["embedding_text"].fillna("")
    print(f"\nEmbedding {len(df)} concept_taxonomy rows...")
    embeddings = embed_texts(model, df["embedding_text"].tolist())

    rows = [
        (
            r.canonical_concept_id, r.canonical_concept_name, r.raw_concept_name,
            r.parent_domain, r.competency_area, r.source_file, r.source_id,
            r.alias_name, bool(r.is_canonical_label), r.embedding_text,
            emb.tolist(),
        )
        for r, emb in zip(df.itertuples(index=False), embeddings)
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO concept_taxonomy (
                canonical_concept_id, canonical_concept_name, raw_concept_name,
                parent_domain, competency_area, source_file, source_id,
                alias_name, is_canonical_label, embedding_text, embedding
            ) VALUES %s
            ON CONFLICT (canonical_concept_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_text = EXCLUDED.embedding_text
            """,
            rows,
        )
    conn.commit()
    print(f"Loaded {len(rows)} rows into concept_taxonomy.")


def load_courses(conn, model, csv_path):
    df = pd.read_csv(csv_path)
    df["embedding_text"] = df["embedding_text"].fillna("")
    print(f"\nEmbedding {len(df)} course rows...")
    embeddings = embed_texts(model, df["embedding_text"].tolist())

    rows = [
        (
            r.course_id, r.source_platform, r.name, r.description,
            r.provider_organization, r.level, r.duration_minutes,
            r.start_date, r.end_date, r.enrollment_type, r.status,
            r.course_url, r.internal_category, r.source, r.source_url,
            r.embedding_text, bool(r.embedding_text_is_description_based),
            emb.tolist(),
        )
        for r, emb in zip(df.itertuples(index=False), embeddings)
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO courses (
                course_id, source_platform, name, description,
                provider_organization, level, duration_minutes,
                start_date, end_date, enrollment_type, status,
                course_url, internal_category, source, source_url,
                embedding_text, embedding_text_is_description_based, embedding
            ) VALUES %s
            ON CONFLICT (course_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_text = EXCLUDED.embedding_text
            """,
            rows,
        )
    conn.commit()
    print(f"Loaded {len(rows)} rows into courses.")


def load_document_chunks(conn, model, chunks_csv_path, domain_tags_csv_path=None):
    df = pd.read_csv(chunks_csv_path)
    df["chunk_text"] = df["chunk_text"].fillna("")
    print(f"\nEmbedding {len(df)} document_chunks rows (this is the largest table, "
          f"may take a few minutes)...")
    embeddings = embed_texts(model, df["chunk_text"].tolist())

    rows = [
        (r.chunk_id, r.parent_doc_id, r.chunk_text, int(r.chunk_order),
         str(r.page_ref) if pd.notna(r.page_ref) else None, emb.tolist())
        for r, emb in zip(df.itertuples(index=False), embeddings)
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO document_chunks (
                chunk_id, parent_doc_id, chunk_text, chunk_order, page_ref, embedding
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
            """,
            rows,
        )
    conn.commit()
    print(f"Loaded {len(rows)} rows into document_chunks.")

    if domain_tags_csv_path and os.path.exists(domain_tags_csv_path):
        tags_df = pd.read_csv(domain_tags_csv_path)
        tag_rows = list(tags_df.itertuples(index=False, name=None))
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO chunk_domain_tags (chunk_id, domain)
                VALUES %s
                ON CONFLICT (chunk_id, domain) DO NOTHING
                """,
                tag_rows,
            )
        conn.commit()
        print(f"Loaded {len(tag_rows)} rows into chunk_domain_tags.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                         help="Directory containing the *_embedready.csv files")
    args = parser.parse_args()

    d = args.data_dir
    concept_csv = os.path.join(d, "concept_taxonomy_embedready.csv")
    courses_csv = os.path.join(d, "courses_master_embedready.csv")
    chunks_csv = os.path.join(d, "document_chunks_embedready.csv")
    domain_tags_csv = os.path.join(d, "chunk_domain_tags.csv")

    for path in (concept_csv, courses_csv, chunks_csv):
        if not os.path.exists(path):
            sys.exit(f"Missing required file: {path}")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    conn = get_connection()
    try:
        load_concept_taxonomy(conn, model, concept_csv)
        load_courses(conn, model, courses_csv)
        load_document_chunks(conn, model, chunks_csv, domain_tags_csv)
    finally:
        conn.close()

    print("\nAll tables embedded and loaded successfully.")


if __name__ == "__main__":
    main()
