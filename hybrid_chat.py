import json
import re
from typing import List
import cohere
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

EMBED_MODEL = "embed-english-light-v3.0"
CHAT_MODEL = "command-r-plus-08-2024"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

try:
    client = cohere.Client(config.COHERE_API_KEY)
    print("✅ Cohere client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Cohere client: {e}")
    exit(1)

try:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    print("✅ Pinecone client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Pinecone client: {e}")
    exit(1)

try:
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=config.PINECONE_VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Pinecone index '{INDEX_NAME}' created")
    else:
        print(f"✅ Using existing Pinecone index '{INDEX_NAME}'")
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print(f"❌ Failed to connect to Pinecone index: {e}")
    exit(1)

try:
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("RETURN 1")
    print("✅ Neo4j connection established successfully")
except Exception as e:
    print(f"❌ Failed to connect to Neo4j: {e}")
    exit(1)


def embed_text(text: str) -> List[float]:
    """Generates a Cohere embedding for a user query string."""
    try:
        resp = client.embed(texts=[text], model=EMBED_MODEL, input_type="search_query")
        return resp.embeddings[0]
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        raise


def pinecone_query(query_text: str, top_k=TOP_K):
    """Embeds the query and retrieves the top-k most similar vectors from Pinecone."""
    try:
        vec = embed_text(query_text)
        res = index.query(vector=vec, top_k=top_k, include_metadata=True, include_values=False)
        print(f"🔍 Found {len(res['matches'])} relevant matches")
        return res["matches"]
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        raise


def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetches neighboring nodes and relationship types from Neo4j for a list of node IDs."""
    facts = []
    try:
        with driver.session() as session:
            for nid in node_ids:
                q = (
                    "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                    "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                    "m.name AS name, m.type AS type, m.description AS description "
                    "LIMIT 10"
                )
                recs = session.run(q, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_desc": (r["description"] or "")[:400],
                        "labels": r["labels"]
                    })
        print(f"🗺️  Retrieved {len(facts)} graph relationships")
        return facts
    except Exception as e:
        print(f"❌ Error fetching graph context: {e}")
        return []


def build_prompt(user_query, pinecone_matches, graph_facts):
    """Builds a Cohere chat prompt combining Pinecone vector results and Neo4j graph facts."""
    system = (
        "You are a helpful travel assistant. Use the provided semantic search results "
        "and graph facts to answer the user's query briefly and concisely. "
        "Cite node ids when referencing specific places or attractions."
    )

    def _is_placeholder(name: str) -> bool:
        """Returns True if the name looks like a generic auto-generated placeholder."""
        if not name:
            return True
        low = name.lower()
        if re.search(r"\battraction\b", low) and re.search(r"\d+", low):
            return True
        if re.match(r"attraction[_ ]?\d+", low):
            return True
        return False

    vec_context = []
    for m in pinecone_matches:
        meta = m.get("metadata", {}) or {}
        score = m.get("score", None)
        raw_name = (meta.get("name") or "").strip()
        if _is_placeholder(raw_name):
            city = meta.get("city") or meta.get("region") or "Unknown location"
            desc = (meta.get("description") or meta.get("semantic_text") or "").strip()
            desc_snip = (desc[:120] + "...") if desc and len(desc) > 120 else desc
            display_name = f"{city} — {desc_snip}" if desc_snip else city
        else:
            display_name = raw_name

        snippet = f"- id: {m['id']}, name: {display_name}, type: {meta.get('type','')}, score: {score}"
        if meta.get("city"):
            snippet += f", city: {meta.get('city')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"User query: {user_query}\n\n"
            "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
            "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
            "Based on the above, answer the user's question. If helpful, suggest 2–3 concrete itinerary steps or tips and mention node ids for references."}
    ]
    return prompt


def call_chat(prompt_messages):
    """Sends the constructed prompt to Cohere's chat model and returns the response text."""
    chat_history = []
    system_message = ""
    user_message = ""

    for msg in prompt_messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        elif msg["role"] == "user":
            user_message = msg["content"]
        elif msg["role"] == "assistant":
            chat_history.append({"role": "CHATBOT", "message": msg["content"]})

    resp = client.chat(
        model=CHAT_MODEL,
        message=user_message,
        chat_history=chat_history,
        preamble=system_message,
        max_tokens=600,
        temperature=0.2
    )
    return resp.text


def interactive_chat():
    """Runs the main chat loop, taking user questions and printing AI-generated travel answers."""
    print("\n🌏 Hybrid Travel Assistant for Vietnam")
    print("=" * 50)
    print("Ask me about Vietnamese destinations, culture, food, and travel tips!")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            query = input("\n🗣️  Enter your travel question: ").strip()
            if not query or query.lower() in ("exit", "quit"):
                print("\n👋 Thanks for using the travel assistant! Safe travels!")
                break

            print("\n🔍 Searching for relevant information...")

            try:
                matches = pinecone_query(query, top_k=TOP_K)
                if not matches:
                    print("⚠️  No relevant information found. Try a different question.")
                    continue
            except Exception as e:
                print(f"❌ Error searching Pinecone: {e}")
                continue

            try:
                match_ids = [m["id"] for m in matches]
                graph_facts = fetch_graph_context(match_ids)
            except Exception as e:
                print(f"❌ Error fetching graph context: {e}")
                graph_facts = []

            try:
                prompt = build_prompt(query, matches, graph_facts)
                answer = call_chat(prompt)
                print("\n🤖 Assistant Answer")
                print("=" * 50)
                print(answer)
                print("=" * 50)
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                print("Please try rephrasing your question.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the travel assistant!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again or restart the application.")


if __name__ == "__main__":
    try:
        interactive_chat()
    except Exception as e:
        print(f"\n❌ Application error: {e}")
    finally:
        if "driver" in globals():
            driver.close()
            print("\n🔐 Database connections closed.")
