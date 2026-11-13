import json
import re
from typing import List
from flask import Flask, render_template, request, jsonify, send_file
from google import genai
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

app = Flask(__name__)

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# Initialize Gemini
try:
    gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
except Exception as e:
    gemini_client = None
    print(f"⚠️ Gemini init error: {e}")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
except Exception as e:
    pinecone_index = None
    print(f"⚠️ Pinecone client init error: {e}")

# Initialize Neo4j
try:
    neo4j_driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
except Exception as e:
    neo4j_driver = None
    print(f"⚠️ Neo4j driver init error: {e}")


def embed_text(text: str) -> List[float]:
    """Generates a 768-dim Gemini embedding for the user search query."""
    if not gemini_client:
        raise RuntimeError("Gemini client is not initialized.")
    res = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={"output_dimensionality": config.PINECONE_VECTOR_DIM}
    )
    return res.embeddings[0].values


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
    """Generates an answer using Google Gemini Flash."""
    vec_snips = [f"- {m['name']} ({m['city']}, {m['type']}, relevance: {m['score']}%)" for m in matches]
    graph_snips = [f"- ({f['source']}) -[{f['rel']}]-> ({f['target_name']})" for f in graph_facts[:15]]

    content = (
        f"You are an energetic, fun Vietnam travel assistant! Use the following retrieved facts to answer the user.\n\n"
        f"User Query: {user_query}\n\n"
        f"Top Semantic Matches (from Pinecone Vector DB):\n" + "\n".join(vec_snips) + "\n\n"
        f"Knowledge Graph Relationships (from Neo4j):\n" + "\n".join(graph_snips) + "\n\n"
        "Provide an inspiring, structured travel recommendation for the user. "
        "Recommend top destinations, key activities, and practical travel tips."
    )

    if gemini_client:
        try:
            resp = gemini_client.models.generate_content(
                model=CHAT_MODEL,
                contents=content
            )
            return resp.text
        except Exception as e:
            print(f"⚠️ Gemini generate error: {e}")

    # Fallback synthesis if rate limited
    destinations = [f"**{m['name']}** ({m['city']})" for m in matches[:4]]
    dest_list = ", ".join(destinations) if destinations else "Vietnam's top destinations"
    
    rel_highlights = []
    for f in graph_facts[:4]:
        rel_highlights.append(f"• **{f['source'].replace('city_', '').replace('_', ' ').title()}** connects to **{f['target_name']}** via *{f['rel'].replace('_', ' ')}*")
    
    rel_text = "\n".join(rel_highlights) if rel_highlights else "• Direct connections to regional travel hubs."

    return (
        f"### 🌴 Top Recommendations for '{user_query}'\n\n"
        f"Based on **Pinecone vector search**, the top matches are: {dest_list}.\n\n"
        f"#### 🗺️ Knowledge Graph Insights (Neo4j):\n"
        f"{rel_text}\n\n"
        f"#### 💡 Travel Tips:\n"
        f"1. **Best Season:** Central & Northern coast destinations are fantastic between February and August.\n"
        f"2. **Local Transport:** Take high-speed trains or scenic coastal coaches between nearby hubs.\n"
        f"3. **Cuisine:** Don't miss the local street food and fresh seafood specialties in the evening night markets!"
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/graph-viz")
def graph_viz():
    return send_file("neo4j_viz.html")


@app.route("/lib/<path:path>")
def send_lib(path):
    from flask import send_from_directory
    return send_from_directory("lib", path)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
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
    print("🚀 Starting Vietnam Travel Chatbot on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
