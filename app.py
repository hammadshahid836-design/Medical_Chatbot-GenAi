from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf_file, text_split
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Together
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

system_prompt = (
    "You are a certified medical assistant chatbot. "
    "Your goal is to provide accurate, evidence-based, and medically verified information. "
    "Use only the retrieved context as your source. "
    "Never assume or fabricate details. "
    "If the context lacks information, clearly state: 'The retrieved sources do not contain enough data.' "
    "When a user asks a question:"
    "1. First, reason through the problem step-by-step internally (but DO NOT show the reasoning)."
    "2. Then provide a clean, well-structured, user-facing answer under the heading:- Final Answer."
    "Always include: definition, causes, symptoms, diagnosis, treatment, and prevention steps when applicable. "
    "Add medical references in parentheses where possible (e.g., WHO, NIH, Mayo Clinic). "
    "Be concise, structured, and neutral in tone."
    "{context}"
)

load_dotenv()

PINECONE_API_KEY= os.getenv('PINECONE_API_KEY')
TOGETHER_API_KEY= os.getenv('TOGETHER_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY 

embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

index_name = "medibot"


docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2})

llm = Together(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    temperature=0.3,
    max_tokens=1000,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response :", response["answer"])
    return str(response["answer"])

# ------------------------
# Run app
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Render's PORT if available
    app.run(host="0.0.0.0", port=port)
