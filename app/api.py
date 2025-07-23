from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from .chat_manager import ChatManager


class ChatMessage(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    timestamp: str


class ConversationMessage(BaseModel):
    timestamp: str
    question: str
    answer: str

class ConversationInfo(BaseModel):
    id: str
    started: str
    first_question: str
    message_count: int

class ConversationResponse(BaseModel):
    id: str
    messages: List[ConversationMessage]


app = FastAPI(
    title="Retail ChatBot API",
    description="API for retail business analysis chatbot with conversation memory",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


chat_manager = ChatManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the chat system on startup"""
    await chat_manager.initialize()


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Send a message to the chatbot"""
    try:
        response = await chat_manager.process_message(
            question=message.question, conversation_id=message.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations", response_model=List[ConversationInfo])
async def list_conversations():
    """List recent conversations"""
    try:
        return await chat_manager.get_recent_conversations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get a specific conversation history"""
    try:
        messages = await chat_manager.get_conversation(conversation_id)
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationResponse(
            id=conversation_id,
            messages=[ConversationMessage(**msg) for msg in messages]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
