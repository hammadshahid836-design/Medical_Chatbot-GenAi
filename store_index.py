from src.helper import load_pdf_file, text_split
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
# Downloading Embedding model from hugging face
from langchain.embeddings import HuggingFaceEmbeddings
import os



load_dotenv()

PINECONE_API_KEY= os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data = 'C:\\Users\\HP User\\Desktop\Medical_chatbot\\Medical_Chatbot-GenAi\\Data')
text_chunks = text_split(extracted_data)
embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")



pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medibot"

pc.create_index(
    name = index_name,
    dimension = 384,
    metric = "cosine",
    spec = ServerlessSpec(
        cloud = "aws",
        region = "us-east-1"
    )
)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings,
    
    
)