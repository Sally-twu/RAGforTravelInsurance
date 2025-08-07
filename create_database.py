"""
# This script processes a PDF document containing terms and conditions for overseas travel insurance,
# splits it into manageable chunks, stores them in a Chroma vector database.

"""
from dotenv import load_dotenv
load_dotenv()
import os
import shutil
import re
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


DATA_PATH = "data/海外旅行不便險條款.pdf"
CHROMA_DB_PATH = "db_1/chroma_db"
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_documents(path: str) -> list[Document]:
    """Load PDF documents"""
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

def clean_content(text):
    # 移除多餘換行與空格
    text = re.sub(r"\n+", "", text)
    text = re.sub(r"[ \t]+", "", text)
    # 移除句中不必要的換行
    text = re.sub(r"(?<![。！？])\n", " ", text)
    return text.strip()

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks, including number, title, content."""
    documents_onepage = ""
    for doc in documents:
        documents_onepage += doc.page_content + "\n"
    # 使用正則表達式分割文檔內容 [^\n]+ 表示匹配「直到遇到換行之前的所有文字」        
    pattern = r"(第[一二三四五六七八九十百千萬]+條)\s+([^\n]+)\n([\s\S]*?)(?=\n第[一二三四五六七八九十百千萬]+條\s|$)"
    matches = re.findall(pattern,  documents_onepage)
    chunks = []
    for number, title, content in matches:  # 從 index 1 開始每兩個一組
        number = number.strip()
        title = title.strip()       
        content = clean_content(content)
        chunks.append(Document(page_content=f"{title}:{content}", 
                               metadata={"law_number":number,"law_title": title,"content": content}))

    return chunks 

def save_to_chroma(chunks: list[Document]) -> Chroma:
    """Create a vector store from the documents."""
    # Clear out the database first.
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    # Create a new DB from the documents.
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    try:
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
        print("Chroma DB created successfully.")
    except Exception as e:
        print(f"Error creating Chroma DB: {e}")
        raise

def main():
    # Load documents
    documents = load_documents(DATA_PATH)
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")

    # Split documents into smaller chunks
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Save chunks to Chroma DB
    save_to_chroma(chunks)


if __name__ == "__main__":
    main()

