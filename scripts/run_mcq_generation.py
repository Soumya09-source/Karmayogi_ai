"""
Run MCQ generation over pending document_chunks.

Usage:
    python scripts/run_mcq_generation.py --limit 20
    python scripts/run_mcq_generation.py --limit 5   # small test batch first

Requires:
    - Ollama running locally (`ollama serve`) with the model pulled
      (`ollama pull llama3.1`)
    - Postgres reachable via the same DATABASE_URL the rest of the app uses
    - document_chunks already populated (they are, per data/document_chunks_embedready.csv)
    - concept_taxonomy already populated WITH embeddings (they are)
"""

import argparse

from app.services.mcq_generation import process_pending_chunks


def main():
    parser = argparse.ArgumentParser(description="Generate MCQs from document_chunks via Ollama.")
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max number of chunks to process in this run (default: 10). "
             "Start small (e.g. 5) to sanity-check output before running a large batch.",
    )
    parser.add_argument(
        "--doc-id", type=str, default=None,
        help="Only process chunks from this specific document (parent_doc_id), "
             "e.g. --doc-id MOSPI-SDG04. Lets you build deep, complete coverage "
             "on chosen documents instead of sampling scattered chunks.",
    )
    args = parser.parse_args()
    process_pending_chunks(limit=args.limit, doc_id=args.doc_id)


if __name__ == "__main__":
    main()
