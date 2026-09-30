"""
Export the current tenant's PostgreSQL chunks for evaluation labeling.

Usage:

    python scripts/export_chunk_catalog.py --tenant-id "<TENANT_UUID>" \
        --output evals/chunk_catalog.jsonl

This is read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.db.models import Chunk
from src.db.session import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    db = SessionLocal()

    try:
        rows = (
            db.query(Chunk)
            .filter(Chunk.tenant_id == args.tenant_id)
            .order_by(
                Chunk.document_id,
                Chunk.version,
                Chunk.chunk_index,
            )
            .all()
        )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as file:
            for chunk in rows:
                payload = {
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "tenant_id": str(chunk.tenant_id),
                    "version": chunk.version,
                    "chunk_index": chunk.chunk_index,
                    "chunk_key": (
                        f"{chunk.document_id}:"
                        f"v{chunk.version}:"
                        f"c{chunk.chunk_index}"
                    ),
                    "text": chunk.text,
                    "section_path": chunk.section_path,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "metadata": chunk.document_metadata,
                }

                file.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(f"Exported {len(rows)} chunks to {output}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
