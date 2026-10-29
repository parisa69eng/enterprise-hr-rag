import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_KEY = os.getenv("AVALAI_API_KEY")
BASE_URL = os.getenv(
    "AVALAI_BASE_URL",
    "https://api.avalai.ir/v1",
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def get_embedding_client() -> OpenAI:
    if not API_KEY:
        raise ValueError(
            "AVALAI_API_KEY is missing from .env"
        )

    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def create_embedding(text: str) -> list[float]:
    client = get_embedding_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
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
