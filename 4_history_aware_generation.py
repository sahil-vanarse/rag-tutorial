from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"

# Embeddings (same as ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# LLM (Groq)
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Store conversation history
chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # STEP 1: Rewrite question using history (if exists)
    if chat_history:
        rewrite_messages = [
            SystemMessage(
                content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        rewritten = model.invoke(rewrite_messages)
        search_question = rewritten.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # STEP 2: Retrieve documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = "\n".join(doc.page_content.split("\n")[:2])
        print(f"  Doc {i}: {preview}...")

    # STEP 3: Build prompt (FIXED f-string issue)
    docs_text = "\n".join([f"- {doc.page_content}" for doc in docs])

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{docs_text}

Please provide a clear, helpful answer using only the information from these documents.
If you can't find the answer in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""

    # STEP 4: Ask model
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on provided documents and conversation history."
        ),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # STEP 5: Save history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
