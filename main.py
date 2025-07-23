from retail_rag import RetailRAG


def show_existing_conversations(conversations):
    """Display existing conversations"""
    if not conversations:
        print("📭 No existing conversations found.")
        return None


    conv_ids = sorted(conversations.keys(), reverse=True)
    recent_convs = conv_ids[:5]  

    print("\n📚 Last 5 Conversations:")
    print("-" * 50)
    for i, conv_id in enumerate(recent_convs, 1):
        first_exchange = conversations[conv_id][0]
        preview = (
            first_exchange["question"][:50] + "..."
            if len(first_exchange["question"]) > 50
            else first_exchange["question"]
        )
        print(f"{i}. ID: {conv_id}")
        print(f"   Started: {first_exchange['timestamp']}")
        print(f"   First question: {preview}")
        print("-" * 50)
    return recent_convs


def main():
 
    print("🚀 Initializing Retail RAG System...")
    rag = RetailRAG()

    print("\n" + "=" * 50)
    print("🏪 RETAIL RAG SYSTEM")
    print("=" * 50)


    conversations = rag.load_conversation_history()
    if conversations:
        recent_convs = show_existing_conversations(conversations)
        print("\n1. Start new conversation")
        print("2. Load previous conversation")
        choice = input("\nChoose option (1/2): ").strip()

        if choice == "2":
            conv_num = input("Enter conversation number (1-5): ").strip()
            try:
                idx = int(conv_num) - 1
                if 0 <= idx < len(recent_convs):
                    conv_id = recent_convs[idx]
                    print(f"\n📂 Loading conversation {conv_id}...")

                    if rag.load_existing_conversation(conv_id):
                        print("✅ Previous conversation loaded!")
                        print("Messages will be appended to this conversation.")
                    else:
                        print("❌ Failed to load conversation, starting new one...")
                        rag.set_new_conversation_id()
                else:
                    print("❌ Invalid choice, starting new conversation...")
                    rag.set_new_conversation_id()
            except (ValueError, IndexError):
                print("❌ Invalid input, starting new conversation...")
                rag.set_new_conversation_id()
        else:
            rag.set_new_conversation_id()

    
    print("\nInitializing RAG system...")
    qa_chain = rag.initialize_rag([])

    print("\n💬 Chat Session Started!")
    print(
        "You can ask any question about retail business, market trends, or general topics."
    )
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print(f"\n📝 Conversation ID: {rag.current_conversation_id}")

    try:
        while True:
            question = input("\n❓ You: ").strip()

            # Check for exit commands
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for the conversation!")
                break

            if not question:
                continue

            response = qa_chain(question)
            print("\n🤖 Assistant:", response["result"])

    except KeyboardInterrupt:
        print("\n\n👋 Chat session interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    finally:
        print("\n📁 Saving conversation history...")
        print(f"Conversation ID: {rag.current_conversation_id}")
        print("You can load this conversation next time you start the chat.")


if __name__ == "__main__":
    main()
