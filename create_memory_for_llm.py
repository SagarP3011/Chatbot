# Step 1: Import Required Libraries
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 2: Load Raw PDF(s)
import os

DATA_PATH = "data/"

def load_pdfs(data_path):
    pdf_data = {}  # Dictionary to store extracted pages by PDF filename

    for pdf_file in os.listdir(data_path):
        if pdf_file.endswith(".pdf"):  # Process only PDFs
            pdf_path = os.path.join(data_path, pdf_file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()  # Extract pages as separate documents
            pdf_data[pdf_file] = documents  # Store list of pages under filename

    return pdf_data

pdf_pages = load_pdfs(DATA_PATH)

# Print results
for pdf, pages in pdf_pages.items():
    print(f"{pdf}: {len(pages)} pages loaded")

# Step 3: Create Chunks of Text

def create_chunks(pdf_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_data = {}  # Dictionary to store chunks per PDF

    for pdf_name, pages in pdf_data.items():
        text_chunks = text_splitter.split_documents(pages)  # Chunk pages of each PDF
        chunked_data[pdf_name] = text_chunks  # Store chunks per file

    return chunked_data

pdf_chunks = create_chunks(pdf_pages)  # Apply chunking

# Print chunk details per PDF
for pdf, chunks in pdf_chunks.items():
    print(f"{pdf}: {len(chunks)} chunks created")

# Step 4: Generate Vector Embeddings

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 5: Store Embeddings in FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"

# Flatten all chunks from the dictionary (pdf_chunks)
all_chunks = [chunk for chunks in pdf_chunks.values() for chunk in chunks]

# Convert text chunks into embeddings and store in FAISS
db = FAISS.from_documents(all_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("FAISS database created and saved successfully!")

