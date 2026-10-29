from pathlib import Path

from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "hr_documents"
VECTOR_SIZE = 1536


def get_qdrant_client() -> QdrantClient:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(
    client: QdrantClient,
    vector_size: int = VECTOR_SIZE,
) -> None:
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        existing_size = info.config.params.vectors.size

        if existing_size == vector_size:
            print("Collection already has the correct dimensions.")
            return

        count = client.count(
            collection_name=COLLECTION_NAME
        ).count

        if count > 0:
            raise ValueError(
                "Collection contains data with a different "
                "vector size. Manual migration is required."
            )

        client.delete_collection(COLLECTION_NAME)
        print("Deleted the empty test collection.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"Created collection with {vector_size} dimensions.")


if __name__ == "__main__":
    client = get_qdrant_client()

    try:
        ensure_collection(client)

        info = client.get_collection(COLLECTION_NAME)
        print("Collection:", COLLECTION_NAME)
        print("Vector dimensions:", info.config.params.vectors.size)
        print(
            "Stored points:",
            client.count(collection_name=COLLECTION_NAME).count,
        )
    finally:
        client.close()
