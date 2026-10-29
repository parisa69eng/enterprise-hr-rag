from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import models

from src.chunking import split_markdown_by_sections
from src.embeddings import create_embedding
from src.vector_store import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    ensure_collection,
    get_qdrant_client,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def ingest_documents() -> None:
    document_paths = sorted(DOCUMENTS_DIR.glob("*.md"))

    if not document_paths:
        raise FileNotFoundError(
            f"No Markdown documents found in {DOCUMENTS_DIR}"
        )

    # Split all documents before modifying the database.
    prepared_chunks = []

    for file_path in document_paths:
        markdown_text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        chunks = split_markdown_by_sections(markdown_text)

        for chunk_index, chunk in enumerate(chunks):
            prepared_chunks.append({
                "id": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{file_path.name}:{chunk_index}",
                    )
                ),
                "text": chunk["text"],
                "source": file_path.name,
                "section_title": chunk["section_title"],
                "chunk_index": chunk_index,
            })

        print(f"Chunked {file_path.name}: {len(chunks)} chunks")

    if not prepared_chunks:
        raise ValueError("No chunks were generated.")

    print(f"\nTotal chunks prepared: {len(prepared_chunks)}")
    print("Generating embeddings...")

    # Generate embeddings before deleting existing database points.
    points = []

    for chunk in prepared_chunks:
        vector = create_embedding(chunk["text"])

        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE} dimensions, "
                f"got {len(vector)} for {chunk['source']}"
            )

        points.append(
            models.PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "section_title": chunk["section_title"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

    client = get_qdrant_client()

    try:
        ensure_collection(client)

        # Replace the existing test data with the new chunk-level data.
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter()
            ),
            wait=True,
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        stored_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print(f"\nSuccessfully ingested {len(points)} chunks.")
        print(f"Total stored points: {stored_count}")

    finally:
        client.close()


if __name__ == "__main__":
    ingest_documents()
