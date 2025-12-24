import os
from typing import Dict, Any, Optional, List
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv # 把 api_key 存在環境變數中，然後用 load_dotenv() 讀取
from pydantic import BaseModel, Field 
import pandas as pd
import unicodedata
import re
import time
import logging

logging.basicConfig(
    filename="process.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SYSTEM_PROMPT = """
你是一個專業且有幫助的汽車/機車查詢助理。  
你可以使用搜尋工具來回答問題。  

請遵守以下規則：  
1. 你的任務是搜尋並整理合規的 **汽車或機車車型** 資訊。  
2. 回答時必須提供：  

- 廠牌或廠商
- 中文名稱，不需要列出廠牌或廠商的資訊（包含中英文），有重複出現，請只回覆一次就好，不要重複列出。
- 英文名稱，不需要列出廠牌或廠商的資訊（包含中英文），有重複出現，請只回覆一次就好，不要重複列出。
- 常見名稱（若有多個，請列出），且每一個不需要列出廠牌或廠商的資訊（包含中英文），有重複出現，請只回覆一次就好，不要重複列出。

3. 如果在中文名稱出現過的車型，英文名稱也出現過的車型，以及常見名稱也出現過的車型，請只回覆一次就好，不要重複列出。
4.不需提供車型代碼
                            
5. 請使用繁體中文作答。  
6. 輸出請盡量用條列或結構化方式呈現，方便閱讀。  
範例：
車型代碼：NXC125N  
中文名稱：勁戰三代  
英文名稱：CYGNUS-X (3rd Gen)  
常見名稱：勁戰 125、CYGNUS-X 125、勁戰
            
            """

# BaseModel：Pydantic 提供的基底類別，所有模型都繼承它。
# Field(...)：用來設定欄位的描述、預設值、驗證條件等等
class VehicleInfo(BaseModel): # 繼承 BaseModel
    query: Optional[str] = Field(None, description="查詢的車型，例如 GQR125CD") # 可以為 str 或 None
    chinese_name: Optional[str] = Field(None, description="中文車名")
    english_name: Optional[str] = Field(None, description="英文車名")
    common_names: List[str] = Field(default_factory=list, description="常見名稱清單，可能有多個") # 定義成 list ，因為會有多個常見名稱


def create_llm_chain():
    load_dotenv()

    litellm_api_key=os.getenv("LITELLM_API_KEY")
    litellm_api_base=os.getenv("LITELLM_API_BASE")
    tavily_search_api_key=os.getenv("TAVILY_SEARCH_API_KEY")

    llm = ChatOpenAI(
        model='azure-gpt-4.1',
        api_key=litellm_api_key,
        base_url=litellm_api_base,
        temperature=0.7,  # 降低 temperature 以減少觸發內容過濾的機率
        top_p=0.8,
    )

    tools = [
        TavilySearch(max_results=10,
                    tavily_api_key=tavily_search_api_key)
    ]

    llm_with_tool = llm.bind_tools(tools)
    structured_llm = llm_with_tool.with_structured_output(VehicleInfo)

    def chatbot(state: MessagesState):
        # 加入 system prompt
        if not any(isinstance(msg, SystemMessage) for msg in state["messages"]):
            system_message = SystemMessage(content=SYSTEM_PROMPT)
            state["messages"].insert(0, system_message)
        
        messages = llm_with_tool.invoke(state["messages"])
        return {"messages": messages}

    graph_builder = StateGraph(state_schema=MessagesState)
    tool_node = ToolNode(tools)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node('tools', tool_node)

    graph_builder.add_edge(START, "chatbot") # Start the conversation with the chatbot
    graph_builder.add_conditional_edges("chatbot", tools_condition) # Add conditional edges for tool usage(讓llm自行決定判斷條件)
    graph_builder.add_edge("tools", "chatbot") # Allow the tools to call the chatbot 

    return {
        "graph": graph_builder.compile(),
        "structured_llm": structured_llm
    }


def build_car_kind_search_prompt(row: Dict[str, Any]) -> str:
    return f"幫我搜尋以下車子的車型資訊：\n- 廠牌：{row['CAR_MODEL']} - 車型：{row['CAR_KIND']}"

def process_vehicle_batch(
    data: List[Dict[str, Any]],
    graph: Any,
    structured_llm: Any,
    config: Dict[str, Any] = {"configurable": {"thread_id": "1", "user_id": "1"}}
) -> List[VehicleInfo]:
    results = []
    structured_result = []

    for row in data:
        try:
            prompt = build_car_kind_search_prompt(row)
            response = graph.invoke(MessagesState(messages=[prompt]), config=config)
            raw_response = response['messages'][-1].content
            results.append(raw_response)
            vehicle_info = structured_llm.invoke(raw_response)     
            structured_result.append(vehicle_info)
        except Exception as e:
            logging.error(f"Error processing row {row}: {e}")

    return results, structured_result

def normalize_car_kind(s: str) -> str:
    """
    標準化車型名稱：全半形、去除髒字元、壓縮空白、轉小寫
    """
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    illegal_chars = ['*', ':', '@', '#']
    s = re.sub(r'[' + ''.join(re.escape(char) for char in illegal_chars) + ']', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    match = re.fullmatch(r'[\u4E00-\u9FFF]+\s?([A-Za-z0-9]+)\s?[\u4E00-\u9FFF]+', s)
    if match:
        s = match.group(1) 
        
    return s

def _is_weird_car_kind(s):
    raw = s.strip()
    # 直接標記常見無效值
    if raw in ("", "無", "原無記載"):
        return True
    # 全字串皆為非英數及非中文 (符號/標點)
    if re.fullmatch(r"[^0-9A-Za-z\u4e00-\u9fff]+", raw):
        return True
    # 長度為 1 且不是英數或中文
    if len(raw) == 1 and not re.match(r"[0-9A-Za-z\u4e00-\u9fff]", raw):
        return True
    return False

def run_vehicle_search(
    input_file: str, 
    start_idx: Optional[int],
    end_idx: Optional[int] = None) -> None:
    
    chain = create_llm_chain()
    df = pd.read_excel(input_file)
    df = df.iloc[start_idx:end_idx] if start_idx is not None and end_idx is not None else df
    df['CAR_KIND'] = df['CAR_KIND'].astype(str).apply(normalize_car_kind)
    df = df[~df['CAR_KIND'].apply(_is_weird_car_kind)]

    results, structured_result = process_vehicle_batch(
        df.to_dict('records'),  
        chain['graph'],
        chain['structured_llm']
    )

    results_df = pd.DataFrame(results, columns=['raw_response'])
    structured_result_df = pd.DataFrame([
        {
            **row,
            'chinese_name': result.chinese_name,
            'english_name': result.english_name,
            'common_names': result.common_names,
            'alias_kind_name': [name for name in ([result.chinese_name, result.english_name] + result.common_names) if name is not None and name != '']  # 合併非空值
        }
        for row, result in zip(df.to_dict('records'), structured_result)
    ])  

    structured_result_df['alias_kind_name'] = structured_result_df['alias_kind_name'].apply(lambda x: list(set(x)) if x else [])
    final_df = pd.concat([results_df.reset_index(drop=True), structured_result_df.reset_index(drop=True)], axis=1)

    return final_df

if __name__ == "__main__":
    # 建立專案根目錄相對路徑，避免執行位置差異導致找不到檔案
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(data_dir, 'output')

    ocr_path = os.path.join(data_dir, 'ocr_may_5057.xlsx')  # 2026/05 續保件
    ai_cache_path = os.path.join(output_dir, 'ai_search_result.csv')
    ai_cache_df = pd.read_csv(ai_cache_path) if os.path.exists(ai_cache_path) else pd.DataFrame()

    start_time = time.perf_counter()
    logging.info("START AI SEARCH PROCESS")

    # total_records = pd.read_excel(ocr_path).shape[0]
    total_records = 5060
    batch_size = 100  # 每批處理 10 筆資料
    for start_idx in range(5000, total_records, batch_size):
        end_idx = min(start_idx + batch_size, total_records)
        logging.info(f"START NO.{start_idx} TO NO.{end_idx} DATA PROCESSING")    

        final_df = run_vehicle_search(ocr_path, start_idx, end_idx)
        ai_cache_df = pd.concat([ai_cache_df, final_df], ignore_index=True)
        ai_cache_df.drop_duplicates(subset=['CAR_KIND','CAR_MODEL'], keep='last', inplace=True)
        ai_cache_df.to_csv(ai_cache_path, index=False)

        logging.info(f"END NO.{start_idx} TO NO.{end_idx} DATA PROCESSING")


    end_time = time.perf_counter()
    logging.info(f"Total execution time: {end_time - start_time:.2f} s")
    