import json
import time
from tqdm import tqdm
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import config

DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 768 for models/text-embedding-004

genai.configure(api_key=config.GEMINI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

existing_indexes = pc.list_indexes().names()
if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != VECTOR_DIM:
        print(f"Recreating index '{INDEX_NAME}' with dimension {VECTOR_DIM} (was {desc.dimension})...")
        pc.delete_index(INDEX_NAME)
        time.sleep(2)
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index '{INDEX_NAME}' exists with matching dimension ({VECTOR_DIM}).")
else:
    print(f"Creating index: {INDEX_NAME} (dim={VECTOR_DIM})")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


def get_embeddings(texts, model="models/text-embedding-004"):
    """Generates Gemini embeddings for a batch of texts."""
    result = genai.embed_content(
        model=model,
        content=texts,
        task_type="retrieval_document"
    )
    return result["embedding"]


def chunked(iterable, n):
    """Splits an iterable into chunks of size n."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Upserting {len(items)} items to Pinecone using Gemini embeddings...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)
        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]
        index.upsert(vectors=vectors)
        time.sleep(0.1)

    print("All items uploaded successfully to Pinecone.")


if __name__ == "__main__":
    main()
