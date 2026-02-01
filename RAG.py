import os
import torch  #     
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore_from_pdfs(pdf_folder: str, save_path: str):
    all_docs = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Working on: {device.upper()} ---")

    # Load all PDFs in the folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs = loader.load()
            all_docs.extend(docs)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # Use HuggingFace embeddings with GPU support
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': device}  # هذا السطر هو الذي يفعّل كرت الشاشة
    )

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"✅ Vector store saved to: {save_path}")

if __name__ == "__main__":
    build_vectorstore_from_pdfs("DATA", "vectorstore")