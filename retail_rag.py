import os
import json
from datetime import datetime
from typing import Dict, List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


class RetailRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.llm = OllamaLLM(model="mistral:7b")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.conversation_store = "conversations.json"
        self.current_conversation_id = None
        self.set_new_conversation_id()

    def list_conversations(self) -> List[str]:
        if os.path.exists(self.conversation_store):
            with open(self.conversation_store, "r") as f:
                conversations = json.load(f)
            return list(conversations.keys())
        return []

    def set_new_conversation_id(self):
        self.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def set_conversation_id(self, conversation_id: str):
        self.current_conversation_id = conversation_id

    def load_conversation_history(self) -> Dict:
        if os.path.exists(self.conversation_store):
            with open(self.conversation_store, "r") as f:
                return json.load(f)
        return {}

    def load_existing_conversation(self, conversation_id: str) -> bool:
        conversations = self.load_conversation_history()
        if conversation_id in conversations:
            
            self.set_conversation_id(conversation_id)

            
            for exchange in conversations[conversation_id]:
                if "question" in exchange and "answer" in exchange:
                    self.memory.chat_memory.add_user_message(exchange["question"])
                    self.memory.chat_memory.add_ai_message(exchange["answer"])

            print(
                f"✅ Loaded conversation {conversation_id} with {len(conversations[conversation_id])} messages"
            )
            return True
        return False

    def save_conversation(self, question: str, answer: str):
        conversations = self.load_conversation_history()
        if self.current_conversation_id not in conversations:
            conversations[self.current_conversation_id] = []
        conversations[self.current_conversation_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
            }
        )
        with open(self.conversation_store, "w") as f:
            json.dump(conversations, f, indent=2)

    def load_pdf_data(self, pdf_files):
        documents = []
        for pdf_path in pdf_files:
            try:
                loader = PyMuPDFLoader(pdf_path)
                documents.extend(loader.load())
                print(f"✅ Loaded: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"❌ Error loading {os.path.basename(pdf_path)}: {str(e)}")
        return documents

    def process_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        return splits

    def setup_vectorstore(self, splits):
        persist_dir = "./retail_data_store"
        index_name = "retail_faiss_index"

        if os.path.exists(os.path.join(persist_dir, f"{index_name}.faiss")):
            print("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                folder_path=persist_dir,
                index_name=index_name,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print("Creating new FAISS index and embedding documents...")
            vectorstore = FAISS.from_documents(
                documents=splits, embedding=self.embeddings
            )

            os.makedirs(persist_dir, exist_ok=True)

            vectorstore.save_local(folder_path=persist_dir, index_name=index_name)

        return vectorstore

    def check_question_relevance(self, question, vectorstore):
        print("\n🔍 Checking question relevance...")

        relevant_docs = vectorstore.similarity_search_with_score(question, k=5)

        max_distance = 2.0
        scores = [(doc, 1 - (score / max_distance)) for doc, score in relevant_docs]

        RELEVANCE_THRESHOLD = 0.5

        print("\nRelevance Scores:")
        max_score = 0
        for doc, score in scores:
            print(f"- Score: {score:.3f}")
            print(f"- Preview: {doc.page_content[:150]}...\n")
            max_score = max(max_score, score)

        has_relevant_docs = any(score > RELEVANCE_THRESHOLD for _, score in scores)

        if not has_relevant_docs:
            print(
                f"📚 No sufficiently relevant documents found (threshold: {RELEVANCE_THRESHOLD}, best match: {max_score:.3f})"
            )
            print("🤔 Checking if question is retail-related...")

            system_prompt = """You are a retail business expert. Determine if this question is related to:
            - Retail business operations
            - Consumer behavior
            - Market trends
            - Store operations
            - E-commerce
            - Retail technology
            - Supply chain and inventory
            - Retail marketing
            - Customer experience
            
            Respond only with 'true' if related to any of these areas, or 'false' if not."""

            response = self.llm.invoke(
                f"{system_prompt}\n\nQuestion: {question}\nAnswer (true/false):"
            )
            is_retail = response.strip().lower() == "true"
            result = "retail" if is_retail else "general"
            print(f"✨ Question classified as: {result}")
            return result

        print("✅ Found relevant documents in knowledge base")
        return "vectorstore"

    def setup_rag_chain(self, vectorstore):

        retail_template = PromptTemplate(
            template="""You are a retail business expert.
            Previous conversation:
            {chat_history}
            
            Current question: {question}
            
            Please provide expert analysis based on retail industry knowledge, including:
            - Industry best practices
            - Current trends
            - Strategic recommendations
            - Potential challenges and solutions
            
            Assistant:""",
            input_variables=["chat_history", "question"],
        )

        general_template = PromptTemplate(
            template="""You are an AI specifically designed for retail business analysis.
            Previous conversation:
            {chat_history}
            
            Current question: {question}
            
            If the question is outside retail domain, acknowledge this first.
            Then provide a helpful response based on available knowledge.
            
            Assistant:""",
            input_variables=["chat_history", "question"],
        )

    
        vectorstore_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
        )

        def enhanced_qa_chain(question):
            print("\n📝 Processing question:", question)

            relevance = self.check_question_relevance(question, vectorstore)
            print(f"\n🔄 Processing using {relevance.upper()} mode")

            
            chat_history = self.memory.chat_memory.messages
            history_str = ""
            if chat_history:
                history_str = "\n".join(
                    [f"{msg.type}: {msg.content}" for msg in chat_history[-6:]]
                )

            if relevance == "vectorstore":
                print("📊 Using retrieved documents for detailed analysis...")
                response = vectorstore_qa_chain({"question": question})
                answer = response["answer"]

            elif relevance == "retail":
                print("🏪 Using general retail expertise...")
                prompt = retail_template.format(
                    chat_history=history_str, question=question
                )
                answer = self.llm.invoke(prompt)

                
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)

            else:
                print("💡 Answering as general question...")
                prompt = general_template.format(
                    chat_history=history_str, question=question
                )
                answer = self.llm.invoke(prompt)

                
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)

            self.save_conversation(question, answer)

            print("\n✅ Response generated successfully")
            return {"result": answer}

        return enhanced_qa_chain

    def initialize_rag(self, pdf_files):
        if pdf_files:
            documents = self.load_pdf_data(pdf_files)
            splits = self.process_documents(documents)
            vectorstore = self.setup_vectorstore(splits)
        else:
            vectorstore = self.setup_vectorstore([])

        qa_chain = self.setup_rag_chain(vectorstore)
        return qa_chain

    def get_conversation_history(self) -> List[Dict]:
        messages = []
        for msg in self.memory.chat_memory.messages:
            messages.append({"type": msg.type, "content": msg.content})
        return messages

    def clear_memory(self):
        self.memory.clear()
        print("🧹 Conversation memory cleared")

    def save_final_conversation(self):
        """Save the final state of the conversation"""
        conversations = self.load_conversation_history()
        if self.current_conversation_id in conversations:
            with open(self.conversation_store, "w") as f:
                json.dump(conversations, f, indent=2)
            print(
                f"💾 Final conversation saved with ID: {self.current_conversation_id}"
            )
