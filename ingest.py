import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY in your .env file.")

def ingest_data():
    print("1. Loading documents from 'data' folder...")

    loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("Error: No documents found in the 'data' folder. Please add some .txt files and try again.")
        return

    print(f"Total {len(documents)} documents loaded. Now splitting into chunks for better embedding...")
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks. Now generating embeddings and uploading to Pinecone...")


    print("2. Initializing Ollama Embeddings and Pinecone Vector Store...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")


    index_name = "dsa-tutor-local" 
    
    print(f"3. Uploading chunks to Pinecone index '{index_name}'...")
    
    PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=index_name
    )
    
    print("✅ Success! Your DSA notes have been ingested into Pinecone. Now you can ask questions about your notes and get answers based on the content you uploaded!")

if __name__ == "__main__":
    ingest_data()