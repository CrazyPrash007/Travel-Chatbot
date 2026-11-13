import json
import re
from typing import List
from flask import Flask, render_template, request, jsonify, send_file
import cohere
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

app = Flask(__name__)

EMBED_MODEL = "embed-english-light-v3.0"
CHAT_MODEL = "command-r-plus-08-2024"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# Initialize clients
try:
    cohere_client = cohere.Client(config.COHERE_API_KEY)
except Exception as e:
    cohere_client = None
    print(f"⚠️ Cohere client init error: {e}")

try:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=config.PINECONE_VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    pinecone_index = pc.Index(INDEX_NAME)
except Exception as e:
    pinecone_index = None
    print(f"⚠️ Pinecone client init error: {e}")

try:
    neo4j_driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
except Exception as e:
    neo4j_driver = None
    print(f"⚠️ Neo4j driver init error: {e}")


def embed_text(text: str) -> List[float]:
    """Generates a 384-dim Cohere embedding for the user search query."""
    if not cohere_client:
        raise RuntimeError("Cohere client is not configured.")
    resp = cohere_client.embed(texts=[text], model=EMBED_MODEL, input_type="search_query")
    return resp.embeddings[0]


def pinecone_query(query_text: str, top_k=TOP_K):
    """Retrieves top-k similar vectors from Pinecone."""
    if not pinecone_index:
        return []
    vec = embed_text(query_text)
    res = pinecone_index.query(vector=vec, top_k=top_k, include_metadata=True, include_values=False)
    matches = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {}) or {}
        matches.append({
            "id": m.get("id"),
            "score": round(float(m.get("score", 0.0)) * 100, 1),
            "name": meta.get("name") or m.get("id"),
            "type": meta.get("type", "Attraction"),
            "city": meta.get("city") or meta.get("region", "Vietnam"),
            "tags": meta.get("tags", [])
        })
    return matches


def fetch_graph_context(node_ids: List[str]):
    """Retrieves neighboring nodes and relationship facts from Neo4j."""
    facts = []
    if not neo4j_driver:
        return facts
    try:
        with neo4j_driver.session() as session:
            for nid in node_ids:
                q = (
                    "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                    "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                    "m.name AS name, m.type AS type, m.description AS description "
                    "LIMIT 6"
                )
                recs = session.run(q, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "rel": r["rel"],
                        "target_id": r["id"],
                        "target_name": r["name"] or r["id"],
                        "target_type": r["type"] or "Location",
                        "target_desc": (r["description"] or "")[:250]
                    })
    except Exception as e:
        print(f"Error fetching Neo4j graph context: {e}")
    return facts


def generate_llm_answer(user_query, matches, graph_facts):
    """Generates an answer using Cohere command-r-plus, with a resilient fallback if rate-limited."""
    # Build prompt
    system_prompt = (
        "You are an energetic, fun Vietnam travel assistant! "
        "Use the provided semantic search matches and graph knowledge facts to give an inspiring, "
        "helpful, structured travel recommendation. Cite places and give concrete tips."
    )
    
    vec_snips = [f"- {m['name']} ({m['city']}, {m['type']}, relevance: {m['score']}%)" for m in matches]
    graph_snips = [f"- ({f['source']}) -[{f['rel']}]-> ({f['target_name']})" for f in graph_facts[:15]]

    content = (
        f"User Query: {user_query}\n\n"
        f"Top Semantic Matches (from Pinecone):\n" + "\n".join(vec_snips) + "\n\n"
        f"Knowledge Graph Relationships (from Neo4j):\n" + "\n".join(graph_snips) + "\n\n"
        "Provide a helpful travel guide answering the query directly with recommended destinations, "
        "what to do, and travel tips."
    )

    if cohere_client:
        try:
            resp = cohere_client.chat(
                model=CHAT_MODEL,
                message=content,
                preamble=system_prompt,
                max_tokens=650,
                temperature=0.3
            )
            return resp.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print("⚠️ Cohere Trial limit hit (429), generating synthesized answer from Knowledge Graph.")
            else:
                print(f"⚠️ Cohere chat error: {e}")

    # Resilient fallback synthesis if Cohere chat is rate-limited
    destinations = [f"**{m['name']}** ({m['city']})" for m in matches[:4]]
    dest_list = ", ".join(destinations) if destinations else "Vietnam's top destinations"
    
    rel_highlights = []
    for f in graph_facts[:4]:
        rel_highlights.append(f"• **{f['source'].replace('city_', '').replace('_', ' ').title()}** connects to **{f['target_name']}** via *{f['rel'].replace('_', ' ')}*")
    
    rel_text = "\n".join(rel_highlights) if rel_highlights else "• Direct connections to regional travel hubs."

    fallback = (
        f"### 🌴 Top Recommendations for '{user_query}'\n\n"
        f"Based on our **Pinecone vector similarity search**, the most relevant spots are: {dest_list}.\n\n"
        f"#### 🗺️ Knowledge Graph Insights (Neo4j):\n"
        f"{rel_text}\n\n"
        f"#### 💡 Travel Tips:\n"
        f"1. **Best Season:** Central & Northern coast destinations are fantastic between February and August.\n"
        f"2. **Local Transport:** Take high-speed trains or scenic coastal coaches between nearby hubs.\n"
        f"3. **Cuisine:** Don't miss the local street food and fresh seafood specialties in the evening night markets!"
    )
    return fallback


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/graph-viz")
def graph_viz():
    return send_file("neo4j_viz.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        # Step 1: Pinecone Vector Retrieval
        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]

        # Step 2: Neo4j Graph Fact Traversal
        graph_facts = fetch_graph_context(match_ids)

        # Step 3: LLM Generation
        answer = generate_llm_answer(query, matches, graph_facts)

        return jsonify({
            "query": query,
            "answer": answer,
            "matches": matches,
            "graph_facts": graph_facts[:15]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Vietnam Travel Chatbot Frontend on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
