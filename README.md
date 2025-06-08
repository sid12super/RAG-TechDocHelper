# 🧠 RAG-TechDocHelper

**RAG-TechDocHelper** 
is a Retrieval-Augmented Generation (RAG) based tool designed to assist with searching and querying large technical documentation effectively. Built with LangChain, OpenAI, and FAISS, this project lets users embed, store, and retrieve contextually relevant documentation content using natural language queries.

---

## 🚀 Features

- 📚 Ingests and processes large-scale technical documentation
- 🔍 Vector store-powered document retrieval using **FAISS**
- 🤖 Natural language Q&A via **OpenAI GPT**
- 🔗 Built with **LangChain**, modular and extendable
- 💡 CLI interface for ingestion and querying

---

## 🛠️ Tech Stack

- **LangChain** – framework for building LLM-powered applications
- **OpenAI GPT-4/3.5** – question answering and embeddings
- **FAISS** – similarity search and vector store
- **Python** – orchestration and application logic

---

## 📁 Project Structure

RAG-TechDocHelper/

├── src/

│ ├── tech_doc_helper.py # Main logic for ingestion and querying

│ ├── ingest.py # Converts documents into embeddings

│ └── query.py # Interface for querying ingested docs

├── data/ # Raw and processed document sources

├── .env # API keys and config (not tracked in Git)

├── requirements.txt # Python dependencies

└── README.md # You are here!
