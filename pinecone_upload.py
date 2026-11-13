import json
import time
from tqdm import tqdm
import cohere
from pinecone import Pinecone, ServerlessSpec
import config

DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 384 for embed-english-light-v3.0

client = cohere.Client(config.COHERE_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)


def get_embeddings(texts, model="embed-english-light-v3.0"):
    """Generates Cohere embeddings for a list of texts using the search_document input type."""
    resp = client.embed(texts=texts, model=model, input_type="search_document")
    return resp.embeddings


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

    print(f"Upserting {len(items)} items to Pinecone...")

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
        time.sleep(0.2)

    print("All items uploaded successfully.")


if __name__ == "__main__":
    main()
