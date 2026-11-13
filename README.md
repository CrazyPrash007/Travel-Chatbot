# 🌏 Hybrid Travel Chat Assistant

A sophisticated AI-powered travel assistant for Vietnam that combines vector similarity search, graph databases, and large language models (Google Gemini) to provide intelligent travel recommendations.

## 🎯 Overview

This system creates an intelligent chatbot that answers questions about Vietnam travel destinations by:
- **Semantic Search**: Using Pinecone vector database with Google Gemini embeddings (`text-embedding-004`) for contextual similarity
- **Graph Relationships**: Leveraging Neo4j knowledge graph to understand connections between cities, attractions, and activities
- **AI Responses**: Generating natural language recommendations using Google Gemini (`gemini-1.5-flash`)
- **Interactive Visualization**: Cartoon-style responsive Web UI + interactive PyVis network graphs

## 🏗️ Architecture

```
User Query → Gemini Embeddings → Pinecone (Vector Search) → Neo4j (Graph Context) → Gemini Flash LLM → Response
                                            ↓
                               Cartoon Web UI / PyVis Graph
```

## 📋 Features

- 🔍 **Semantic Travel Search**: Find destinations based on interests and preferences
- 🗺️ **Relationship Mapping**: Understand how cities, attractions, and activities connect
- 🤖 **Intelligent Recommendations**: Fast, personalized travel suggestions using Google Gemini
- 🎨 **Cartoon Web Frontend**: Neo-brutalist, colorful, interactive dashboard with instant search
- 📊 **Interactive Graphs**: Explore the full knowledge network in your browser
- 🌐 **Rich Dataset**: 360+ Vietnam travel locations with rich metadata

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database (local or Neo4j Aura cloud)
- Google Gemini API key (free tier works)
- Pinecone account (free tier works)

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy `config.example.py` to `config.py` and add your credentials:
```bash
cp config.example.py config.py
```

Edit `config.py`:
```python
NEO4J_URI = "neo4j+ssc://your-instance.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

GEMINI_API_KEY = "your_gemini_api_key"

PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 768  # Gemini text-embedding-004 output dimension
```

### 3. Database Setup (One-time)

#### Load Data to Neo4j
```bash
python load_to_neo4j.py
```

#### Upload Embeddings to Pinecone
```bash
python pinecone_upload.py
```

### 4. Run the Application

#### Option A: Cartoon Web Frontend (Recommended for Demos!)
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

#### Option B: Terminal CLI Chatbot
```bash
python hybrid_chat.py
```

## 🗂️ Project Structure

```
Travel-Chatbot/
├── app.py                  # Flask web server & REST API
├── templates/
│   └── index.html          # Cartoonish Neo-Brutalist UI
├── hybrid_chat.py          # Terminal CLI chat interface
├── load_to_neo4j.py        # Neo4j graph data loader
├── pinecone_upload.py      # Pinecone vector uploader (Gemini embeddings)
├── visualize_graph.py      # PyVis graph visualizer
├── config.example.py       # Configuration template
├── config.py               # Local credentials (gitignored)
├── vietnam_travel_dataset.json  # Travel dataset (360+ locations)
├── requirements.txt        # Python dependencies
├── neo4j_viz.html          # Pre-generated interactive graph
├── lib/                    # JavaScript visualization libraries
└── README.md               # Documentation
```

## 📄 License

This project is for educational and personal use.
