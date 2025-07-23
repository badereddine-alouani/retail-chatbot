# Retail Chatbot with RAG

A conversational AI system for retail business analysis using Retrieval-Augmented Generation (RAG).

## Features
- FastAPI-based chat endpoints
- Conversation memory and history
- RAG system with FAISS vector store
- PDF document processing
- Ollama integration with Mistral-7B

## Setup
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the FastAPI server:
```bash
python run.py
```

## Project Structure
```
retail-chatbot/
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── chat_manager.py
├── retail_data_store/   # Vector store
├── retail_rag.py
├── run.py
├── requirements.txt
└── README.md
```

## API Endpoints
- POST /chat - Send messages
- GET /conversations - List recent conversations
- GET /conversation/{conversation_id} - Get conversation history