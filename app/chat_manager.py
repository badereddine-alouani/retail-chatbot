from typing import Dict, List, Optional
import asyncio
from datetime import datetime
from retail_rag import RetailRAG


class ChatManager:
    def __init__(self):
        self.rag = RetailRAG()
        self.qa_chain = None

    async def initialize(self):
        """Initialize the RAG system"""
        # Run initialization in a thread pool to avoid blocking
        self.qa_chain = await asyncio.get_event_loop().run_in_executor(
            None, self.rag.initialize_rag, []
        )

    async def process_message(
        self, question: str, conversation_id: Optional[str] = None
    ) -> Dict:
        """Process a chat message"""
        # Load existing conversation if ID provided
        if conversation_id:
            self.rag.load_existing_conversation(conversation_id)
        else:
            self.rag.set_new_conversation_id()

        # Get response from RAG system (run in thread pool)
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.qa_chain, question
        )

        timestamp = datetime.now().isoformat()

        return {
            "answer": response["result"],
            "conversation_id": self.rag.current_conversation_id,
            "timestamp": timestamp,
        }

    async def get_recent_conversations(self) -> List[Dict]:
        """Get list of recent conversations"""
        conversations = self.rag.load_conversation_history()
        conv_list = []

        for conv_id, messages in conversations.items():
            if messages:
                conv_list.append(
                    {
                        "id": conv_id,
                        "started": messages[0]["timestamp"],
                        "first_question": messages[0]["question"],
                        "message_count": len(messages),
                    }
                )


        conv_list.sort(key=lambda x: x["started"], reverse=True)
        return conv_list[:5] 

    async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a specific conversation history"""
        conversations = self.rag.load_conversation_history()
        return conversations.get(conversation_id)
