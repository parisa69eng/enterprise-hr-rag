from src.embeddings import create_embedding
from src.vector_store import COLLECTION_NAME, get_qdrant_client


def search_documents(query: str, limit: int = 3) -> None:
    query_vector = create_embedding(query)
    client = get_qdrant_client()

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        if not results.points:
            print("No matching chunks found.")
            return

        for rank, point in enumerate(results.points, start=1):
            payload = point.payload or {}

            print(f"\n--- Result {rank} ---")
            print(f"Source: {payload.get('source', 'Unknown')}")
            print(
                f"Section: "
                f"{payload.get('section_title', 'Unknown')}"
            )
            print(f"Score: {point.score:.4f}")
            print(f"Text:\n{payload.get('text', '')}")

    finally:
        client.close()


if __name__ == "__main__":
    question = input("Ask an HR question: ").strip()

    if question:
        search_documents(question)
    else:
        print("Please enter a question.")
