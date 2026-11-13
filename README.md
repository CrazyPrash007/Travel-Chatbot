# 🌏 Hybrid Travel Chat Assistant

A sophisticated AI-powered travel assistant for Vietnam that combines vector similarity search, graph databases, and large language models to provide intelligent travel recommendations.

## 🎯 Overview

This system creates an intelligent chatbot that can answer questions about Vietnam travel destinations by:
- **Semantic Search**: Using Pinecone vector database with Cohere embeddings for contextual similarity
- **Graph Relationships**: Leveraging Neo4j to understand connections between cities, attractions, and activities
- **AI Responses**: Generating natural language recommendations using Cohere's `command-r-plus` model
- **Interactive Visualization**: Creating network graphs of travel relationships

## 🏗️ Architecture

```
User Query → Cohere Embed → Pinecone (Vector Search) → Neo4j (Graph Context) → Cohere Chat → Response
                                     ↓
                              Graph Visualization (PyVis)
```

## 📋 Features

- 🔍 **Semantic Travel Search**: Find destinations based on interests and preferences
- 🗺️ **Relationship Mapping**: Understand how cities and attractions connect
- 🤖 **Intelligent Recommendations**: Get personalized travel suggestions using Cohere command-r-plus
- 📊 **Interactive Graphs**: Visualize travel networks
- 🌐 **Rich Dataset**: 360+ Vietnam travel locations with detailed metadata

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database (local or Neo4j Aura cloud)
- Cohere API account (free tier works)
- Pinecone account (free tier works)

### 1. Installation

```bash
pip install neo4j cohere pinecone pyvis networkx tqdm python-dotenv
```

### 2. Configuration

Edit `config.py` with your credentials:
```python
NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

COHERE_API_KEY = "your_cohere_api_key"

PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384  # Cohere embed-english-light-v3.0 uses 384 dimensions
```

### 3. Database Setup

#### Load Data to Neo4j (run once)
```bash
python load_to_neo4j.py
```

#### Upload Embeddings to Pinecone (run once)
```bash
python pinecone_upload.py
```

### 4. Run the Chat Assistant

```bash
python hybrid_chat.py
```

## 🗂️ Project Structure

```
Travel-Chatbot/
├── hybrid_chat.py          # Main chat interface
├── load_to_neo4j.py        # Neo4j data loader
├── pinecone_upload.py      # Pinecone vector uploader
├── visualize_graph.py      # Graph visualization generator
├── config.py               # Configuration & API credentials
├── vietnam_travel_dataset.json  # Travel data (360+ locations)
├── requirements.txt        # Python dependencies
├── neo4j_viz.html          # Pre-generated graph visualization
├── lib/                    # JavaScript visualization libraries
└── README.md               # This file
```

## 💾 Data Structure

Each travel entity includes:
```json
{
  "id": "city_hanoi",
  "type": "City",
  "name": "Hanoi",
  "region": "Northern Vietnam",
  "description": "Rich cultural hub with temples and street food...",
  "best_time_to_visit": "February to May",
  "tags": ["culture", "food", "heritage"],
  "semantic_text": "Optimized text for vector search...",
  "connections": [{"relation": "Connected_To", "target": "city_hue"}]
}
```

## 🎮 Usage Examples

### Basic Queries
```
"What are the best beaches in Vietnam?"
"Show me cultural sites near Hanoi"
"I want adventure activities in Northern Vietnam"
"Plan a 3-day itinerary for Ho Chi Minh City"
```

### Advanced Features
- **Graph Visualization**: Run `python visualize_graph.py` to create `neo4j_viz.html`
- **Data Exploration**: Use Neo4j Browser to explore relationships

## 🔧 Configuration

### Key Settings in `hybrid_chat.py`:
- `EMBED_MODEL`: Cohere embedding model (`embed-english-light-v3.0`, 384 dims)
- `CHAT_MODEL`: Cohere chat model (`command-r-plus-08-2024`)
- `TOP_K`: Number of similar results to retrieve (default: 5)

### Cohere Embedding Models:
| Model | Dimensions | Best For |
|-------|------------|----------|
| embed-english-light-v3.0 | 384 | Fast, cost-effective (used here) |
| embed-english-v3.0 | 1024 | Higher accuracy |

## 🐛 Troubleshooting

### Common Issues:

1. **"Module not found" errors**
   ```bash
   pip install neo4j cohere pinecone pyvis networkx tqdm python-dotenv
   ```

2. **Neo4j connection failed**
   - Check NEO4J_URI, credentials in `config.py`
   - Ensure Neo4j Aura instance is running

3. **Pinecone API errors**
   - Verify PINECONE_API_KEY in `config.py`
   - Check index name and dimension (must be 384)

4. **Cohere API errors**
   - Confirm COHERE_API_KEY in `config.py`
   - Check API usage limits

5. **Empty search results**
   - Ensure data is loaded: `python load_to_neo4j.py`
   - Verify vector upload: `python pinecone_upload.py`

## 🚧 Known Limitations

1. **Language**: Currently English-only interface
2. **Geography**: Focused on Vietnam only
3. **Real-time Data**: Static dataset (not live updated)
4. **Concurrent Access**: Single-user design

## 🛣️ Roadmap

- [ ] Web interface with React/Vue.js
- [ ] Multi-language support
- [ ] Real-time data integration
- [ ] User preference learning
- [ ] Mobile app development
- [ ] Integration with booking APIs
- [ ] Advanced itinerary planning

## 📄 License

This project is for educational and personal use. Please ensure compliance with API terms of service for Cohere, Pinecone, and Neo4j.

---

**Happy Traveling! 🌟**

Built with ❤️ using Cohere, Pinecone, Neo4j, and Python