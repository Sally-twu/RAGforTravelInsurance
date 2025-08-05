"""
# This script processes a PDF document containing terms and conditions for overseas travel insurance,
# splits it into manageable chunks, stores them in a Chroma vector database.

"""
import os
from dotenv import load_dotenv
load_dotenv()
from datasets import Dataset
import openai
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from ragas import evaluate
from ragas.metrics import (
   context_precision,
    context_recall,
    answer_relevancy,
    answer_correctness,   
    faithfulness
)


CHROMA_DB_PATH = "db/chroma_db"
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
請根據以下資訊回答問題:{context}
問題: {question}
"""

questions = [
    "什麼情況下可以申請旅遊延誤賠償?",
    "行李遺失後應該如何申請理賠?",
    "哪些原因屬於不可理賠範圍?"
]

ground_truths = [
    "如果因為航空公司延誤或取消航班，導致旅客在機場等待超過4小時，可以申請旅遊延誤賠償。另外，被保險人於海外旅行期間內，其隨行託運並取得託運行李領取單之個人行李因公共交通工具業者之處理失當，致其在抵達目的地六小時後仍未領得時間，亦可申請旅遊延誤賠償。",
    "若是因為竊盜、強盜與搶奪而遺失行李，旅客應該立即向當地警方報案並索取相關證明文件。若是因為航空公司處理失當導致的毀損、滅失或遺失，旅客應該立即向航空公司報告並索取相關證明文件，然後申請理賠。",
    "針對事故不可理賠的範圍為一、被保險人飲酒後駕（騎）車、其吐氣或血液所含酒精成份超過當地道路交通法令規定標準；或因吸食、施打、服用毒品所致之賠償責任；二、懷孕、流產或分娩；三、任何合格醫生已告知被保險人身體狀況不適合旅行，或旅行之目的係為診療或就醫者；四、不需由被保險人負擔費用之服務，或被保險人預定旅程成本中已包含之費用。"
]

# Prepare the DB
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

data_sample = {
    "question": [],
    "answer": [],
    "ground_truth": [],
    "contexts": []
}

for question, ground_truth in zip(questions, ground_truths):
    results = db.similarity_search_with_relevance_scores(question, k=3)
    contexts= [doc.metadata.get('content') for doc, _score in results]
    context_text_str = "\n\n".join([doc.metadata.get('content') for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text_str, question=question)
    
    # Create the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    response_text = llm.invoke(prompt)
    
    data_sample["question"].append(question)
    data_sample["answer"].append(response_text.content.strip())
    data_sample["ground_truth"].append(ground_truth)
    data_sample["contexts"].append(contexts)

dataset = Dataset.from_dict(data_sample)

metrics = [
    context_precision,
    context_recall,
    answer_relevancy,
    answer_correctness,
    faithfulness
]
evaluation_results = evaluate(dataset, metrics=metrics)
print(evaluation_results)


