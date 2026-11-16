
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_embedding_client() -> OpenAI:
    api_key = os.getenv("AVALAI_API_KEY")

    base_url = os.getenv(
        "AVALAI_BASE_URL",
        "https://api.avalai.ir/v1",
    )

    if not api_key:
        raise ValueError(
            "AVALAI_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def create_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small",
    )

    client = get_embedding_client()

    response = client.embeddings.create(
        model=embedding_model,
        input=text.strip(),
    )

    return response.data[0].embedding


if __name__ == "__main__":
    sample_text = (
        "Employees receive 25 days of annual leave "
        "per calendar year."
    )

    vector = create_embedding(sample_text)

    print("Embedding created successfully!")
    print("Vector dimensions:", len(vector))
    print("First five values:", vector[:5])
