"""
# This script processes a PDF document containing terms and conditions for overseas travel insurance,
# splits it into manageable chunks, stores them in a Chroma vector database.

"""
import os
from dotenv import load_dotenv
load_dotenv()
import argparse
import openai
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_DB_PATH = "db/chroma_db"
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
Question: {question}
Answer the question based only on the following context:
{context}
---
"""

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
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return
    
    context_text = "\n\n---\n\n".join([doc.metadata.get("content") for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Create the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    response_text = llm.invoke(prompt)

    sources = [
        doc.metadata.get('law_number', '') + " " + doc.metadata.get('law_title', '')
        for doc, _score in results
    ]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
 main()
