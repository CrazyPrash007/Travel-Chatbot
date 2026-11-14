# 🌴 Vietnam Travel Explorer - Hybrid RAG & Knowledge Graph

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash%20Lite-orange.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless%20Vector%20DB-blueviolet.svg?logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Aura%20Knowledge%20Graph-008CC1.svg?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent travel recommendation engine and interactive knowledge exploration system for Vietnam. This project pairs **semantic vector retrieval (Pinecone)** with a **topological knowledge graph (Neo4j Aura)** and **Google Gemini** to synthesize structured, hallucination-free travel itineraries.

It features a high-contrast **Neo-Brutalist cartoon web dashboard** and a self-contained **360-node interactive knowledge galaxy** with spring-physics simulation.

---

## 🏗️ System Architecture

```
                       +-----------------------------------+
                       |      User Query (Web / CLI)       |
                       +-----------------+-----------------+
                                         |
                       +-----------------v-----------------+
                       |    Gemini Text Embeddings (768d)  |
                       +-----------------+-----------------+
                                         |
                +------------------------+------------------------+
                |                                                 |
+---------------v---------------+                 +---------------v---------------+
|    Pinecone Vector Database   |                 |       Neo4j Knowledge Graph   |
|   (Cosine Semantic Similarity)|                 | (Entity Relationships & Hubs) |
+---------------+---------------+                 +---------------+---------------+
                |                                                 |
                | Top 5 Matches                                   | Multi-Hop Context
                +------------------------+------------------------+
                                         |
                       +-----------------v-----------------+
                       |   Prompt Synthesis & Sanitizer    |
                       +-----------------+-----------------+
                                         |
                       +-----------------v-----------------+
                       |   Google Gemini 3.5 Flash Lite    |
                       +-----------------+-----------------+
                                         |
                       +-----------------v-----------------+
                       | Cartoon Web UI / Graph Explorer   |
                       +-----------------------------------+
```

---

## ✨ Key Features

- **Hybrid RAG Pipeline**: Merges unstructured semantic similarity (768-dimensional vector embeddings) with structured relational facts (cities, sights, hotels, activities).
- **Neo4j Knowledge Graph Galaxy**:
  - **360 Nodes & 370 Relationships**: 10 major city hubs, 150 attractions, 100 hotels, and 100 activities.
  - **Spring-Physics Simulation**: Drag nodes to feel rubbery, tactile force-directed physics.
  - **Live Filter HUD**: Real-time autocomplete search, city hub jump selector, category filter pills (`Cities`, `Attractions`, `Hotels`, `Activities`), and physics toggles (`Freeze`, `Float`, `Jelly Wobble`).
  - **Real-Time Percentage Progress Loader**: Visual indicator (`0%` to `100%`) with candy-striped progress bar and stage descriptions.
- **Cartoon Neo-Brutalist Web UI**:
  - Retro high-contrast visual design with bold borders, drop shadows, and lively color accents.
  - Instant suggestion pills for one-click discovery.
  - Deep linking from graph nodes directly into the chatbot without page reloads.
- **Sanitized Output Pipeline**: Eliminates artificial conversational filler, robotic disclaimers, and typographic markers, returning concise recommendations.
- **Cloud-Ready & Cost-Free**: Configured to run on free-tier infrastructure (Render, Pinecone Serverless, Neo4j Aura Free, and Gemini API Free Tier).

---

## 🛠️ Technology Stack

| Layer | Component | Version / Model | Description |
| :--- | :--- | :--- | :--- |
| **LLM & Reasoning** | Google Gemini | `gemini-3.5-flash-lite` | Generates natural language itineraries from retrieved context |
| **Embeddings** | Google Gemini | `gemini-embedding-001` | Generates 768-dimensional dense vector embeddings |
| **Vector Database** | Pinecone | Serverless (AWS `us-east-1`) | Fast top-k approximate nearest neighbor cosine search |
| **Graph Database** | Neo4j AuraDB | Neo4j 5+ (Cloud TLS) | Graph traversal for spatial, regional, and category relationships |
| **Backend & API** | Flask / Gunicorn | Python 3.10+ | Lightweight REST API and web application server |
| **Interactive Graph** | Vis-Network | v9.1.2 | Hardware-accelerated canvas visualization with forceAtlas2 physics |
| **Styling & Theme** | Neo-Brutalist CSS | Fredoka + Plus Jakarta Sans | Custom cartoon design with bold drop shadows and responsive layout |

---

## 🗂️ Project Structure

```
Travel-Chatbot/
├── app.py                      # Flask web application and REST API endpoints
├── hybrid_chat.py              # Interactive terminal CLI chat interface
├── visualize_graph.py          # Script to generate interactive 360-node HTML graph
├── neo4j_viz.html              # Standalone interactive knowledge graph visualization
├── load_to_neo4j.py            # One-time data loader for Neo4j Aura knowledge graph
├── pinecone_upload.py          # One-time batch vector embedding and Pinecone uploader
├── vietnam_travel_dataset.json # Curated travel dataset (360 verified locations)
├── templates/
│   └── index.html              # Cartoon Neo-Brutalist web dashboard
├── lib/
│   ├── vis-9.1.2/              # Local Vis-Network JavaScript and CSS bundles
│   ├── bindings/               # PyVis JavaScript bindings
│   └── tom-select/             # Select and search UI assets
├── Procfile                    # Cloud process definition for Render / Railway
├── requirements.txt            # Python dependencies
├── config.example.py           # Configuration template with placeholders
├── config.py                   # Local credentials file (gitignored)
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Neo4j Aura Account** (Free tier at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/))
- **Pinecone Account** (Free tier at [pinecone.io](https://www.pinecone.io/))
- **Google Gemini API Key** (Free tier at [aistudio.google.com](https://aistudio.google.com/))

---

### Step 1: Clone & Install Dependencies

```bash
git clone https://github.com/CrazyPrash007/Travel-Chatbot.git
cd Travel-Chatbot
pip install -r requirements.txt
```

---

### Step 2: Configure Environment Variables

You can configure credentials either by creating `config.py` locally or by exporting environment variables.

#### Option A: Local `config.py`
Copy `config.example.py` to `config.py`:
```bash
cp config.example.py config.py
```

Edit `config.py` with your credentials:
```python
NEO4J_URI = "neo4j+ssc://your-instance.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"

GEMINI_API_KEY = "your_google_gemini_api_key"

PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 768
```

#### Option B: Environment Variables (Recommended for Production)
```bash
export NEO4J_URI="neo4j+ssc://your-instance.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_neo4j_password"
export GEMINI_API_KEY="your_google_gemini_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
export PINECONE_INDEX_NAME="vietnam-travel"
export PINECONE_VECTOR_DIM="768"
```

---

### Step 3: Seed Databases (One-Time Setup)

#### 1. Seed Knowledge Graph in Neo4j:
```bash
python load_to_neo4j.py
```
*Creates 360 Entity nodes (`City`, `Attraction`, `Hotel`, `Activity`) and 370 relationships (`Connected_To`, `Located_In`, `Available_In`).*

#### 2. Seed Vector Embeddings in Pinecone:
```bash
python pinecone_upload.py
```
*Generates 768-dim embeddings via `gemini-embedding-001` with exponential backoff and uploads 360 vectors with metadata.*

#### 3. Generate Knowledge Graph HTML:
```bash
python visualize_graph.py
```
*Builds the self-contained `neo4j_viz.html` with real-time percentage progress loader.*

---

### Step 4: Run the Application

#### Option A: Cartoon Web Dashboard (Recommended)
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.
- **Main Chat Dashboard**: `http://localhost:5000`
- **3D Knowledge Graph**: `http://localhost:5000/graph-viz`

#### Option B: Terminal CLI Interface
```bash
python hybrid_chat.py
```

---

## 🌐 Free Cloud Deployment (Render.com)

Deploy this application online at zero cost:

1. Push your repository to GitHub.
2. Sign up on **[render.com](https://render.com/)** and click **New + > Web Service**.
3. Select your repository `Travel-Chatbot`.
4. Configure service parameters:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Free` ($0/month)
5. Under **Environment Variables**, add:
   - `NEO4J_URI`: `neo4j+ssc://your-instance.databases.neo4j.io`
   - `NEO4J_USER`: `neo4j`
   - `NEO4J_PASSWORD`: `your_password`
   - `GEMINI_API_KEY`: `your_gemini_api_key`
   - `PINECONE_API_KEY`: `your_pinecone_api_key`
   - `PINECONE_INDEX_NAME`: `vietnam-travel`
   - `PINECONE_VECTOR_DIM`: `768`
6. Click **Deploy Web Service**. You will receive an SSL-enabled public URL (e.g. `https://vietnam-travel-chatbot.onrender.com`).

---

## 📡 API Reference

### Chat Completion Endpoint

- **Endpoint**: `POST /api/chat`
- **Content-Type**: `application/json`

#### Request Body
```json
{
  "query": "What are the best beaches in Vietnam?"
}
```

#### Response Body
```json
{
  "query": "What are the best beaches in Vietnam?",
  "answer": "Vietnam offers diverse coastal experiences across its central and southern coastlines...",
  "matches": [
    {
      "id": "city_ha_long",
      "name": "Ha Long Bay",
      "city": "Northern Vietnam",
      "type": "City",
      "score": 62.5,
      "tags": ["beach", "cruise", "nature"]
    }
  ],
  "graph_facts": [
    {
      "source": "city_ha_long",
      "rel": "Connected_To",
      "target_id": "city_hoi_an",
      "target_name": "Hoi An",
      "target_type": "City",
      "target_desc": "Hoi An is located in Central Vietnam..."
    }
  ]
}
```

---

## 🔒 Security & Privacy

- **Git-Ignored Secrets**: `config.py` is excluded by `.gitignore` and is never committed to source control.
- **Environment Isolation**: Production environments use injected environment variables.
- **Sanitized Inputs & Outputs**: Strict escaping and model output cleanup prevents prompt injection and text distortion.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
