from flask import Flask, render_template, request
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
from src.prompt import system_prompt
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(PINECONE_API_KEY)
print(GEMINI_API_KEY)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    GEMINI_API_KEY=GEMINI_API_KEY  # from aistudio.google.com
)

embeddings = download_embeddings()

index_name = "my-medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding=embeddings
)

# Define retriever as a tool (new LangChain 1.0 way)
@tool(response_format="content_and_artifact")
def retrieve_medical_context(query: str):
    """Retrieve relevant medical information to help answer a query."""
    retrieved_docs = docsearch.similarity_search(query, k=3)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Create agent (replaces create_retrieval_chain + create_stuff_documents_chain)
agent = create_agent(
    chat_model,
    tools=[retrieve_medical_context],
    system_prompt=system_prompt
)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = agent.invoke({
        "messages": [{"role": "user", "content": msg}]
    })
    answer = response["messages"][-1].content
    print("Response : ", answer)
    return str(answer[0]["text"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8000, debug = True)
