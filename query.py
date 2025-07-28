"""
# This script processes a PDF document containing terms and conditions for overseas travel insurance,
# splits it into manageable chunks, stores them in a Chroma vector database.

"""
import os
from dotenv import load_dotenv
load_dotenv()
import argparse
import openai
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_DB_PATH = "db/chroma_db"
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()  
    query_text = args.query_text

    # Prepare the DB
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

    # Search the DB
    result = db.similarity_search(query_text, k=3)
    print(f"Query: {query_text}")
    print(f"Results:{result}")       


if __name__ == "__main__":
 main()

"""
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
    )"""