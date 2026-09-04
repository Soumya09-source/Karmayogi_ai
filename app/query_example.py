"""
query_example.py

Demonstrates the actual recommendation-engine query from the
architecture: given an employee's weak concept, find the most
semantically similar courses via pgvector cosine similarity.

Usage:
    python query_example.py "Stratified Sampling"
"""

import os
import sys

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "all-MiniLM-L6-v2"


def get_connection():
    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ.get("PGPORT", 5432),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )
    register_vector(conn)
    return conn


def find_top_courses_for_concept(conn, model, concept_text, top_n=5):
    query_embedding = model.encode(concept_text, normalize_embeddings=True)

    # Cosine distance operator is `<=>` in pgvector. Since embeddings are
    # normalized, `1 - (embedding <=> query)` gives cosine similarity.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT course_id, name, provider_organization,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM courses
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding.tolist(), query_embedding.tolist(), top_n),
        )
        return cur.fetchall()


def find_top_chunks_for_concept(conn, model, concept_text, top_n=5):
    query_embedding = model.encode(concept_text, normalize_embeddings=True)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, parent_doc_id,
                   left(chunk_text, 150) AS preview,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding.tolist(), query_embedding.tolist(), top_n),
        )
        return cur.fetchall()


def main():
    concept_text = sys.argv[1] if len(sys.argv) > 1 else "Stratified Sampling"
    print(f"Query concept: {concept_text!r}\n")

    model = SentenceTransformer(MODEL_NAME)
    conn = get_connection()

    print("Top matching courses:")
    for course_id, name, provider, sim in find_top_courses_for_concept(conn, model, concept_text):
        print(f"  [{sim:.3f}] {name}  ({provider})  -- {course_id}")

    print("\nTop matching document chunks:")
    for chunk_id, parent_doc, preview, sim in find_top_chunks_for_concept(conn, model, concept_text):
        print(f"  [{sim:.3f}] {chunk_id} ({parent_doc}): {preview}...")

    conn.close()


if __name__ == "__main__":
    main()
