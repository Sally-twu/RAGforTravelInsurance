#import onnxruntime
#print(onnxruntime.__version__)

import re
import pandas as pd

# 範例法條內容
text = """
第七十九條 特別不保事項（事故） 
對於下列事故，本公司不負理賠責任：
一、被保險人飲酒後駕（騎）車、其吐氣或血液所含酒精成份超過當地道路交通法令規定標準；
或因吸食、施打、服用毒品所致之賠償責任。
二、懷孕、流產或分娩。但其併發症，或因意外傷害或病因性所致之流產、分娩，不在此限。
三、任何合格醫生已告知被保險人身體狀況不適合旅行，或旅行之目的係為診療或就醫者。
四、不需由被保險人負擔費用之服務，或被保險人預定旅程成本中已包含之費用。
前項第二款所稱之病因性所致之流產、分娩，係指葡萄胎、過期流產、子宮外孕、迫切流產、
先兆性流產、前置胎盤、胎盤早期剝離、子癇前症、子癇症、妊娠期之過度嘔吐、妊娠毒血症、
妊娠期末稍神經炎等妊娠併發症所致之流產、分娩

第八十一條 支付保險金之方式 
本保險承保之項目，如為本公司簽約之救援服務公司或被保險人親屬先行墊付並出具支付證明
者，本公司得直接向該公司或親屬給付保險金。
若先行墊付者非屬前項約定之人，則以被保險人或被保險人之法定繼承人為給付對象。
"""

# -------- 萃取法條編號與標題 --------
title_match = re.match(r"^(第[一二三四五六七八九十百千萬]+條)\s*(.+)", text.strip())
article_number = title_match.group(1)
article_title = title_match.group(2)

# -------- 去除標題段落 --------
text_no_title = re.sub(r"^第[一二三四五六七八九十百千萬]+條\s*(.+)", "", text, flags=re.MULTILINE)

# -------- 抓主文（直到第一個條列符號為止） --------
match = re.search(r"^(.*?)(?=\s*[一二三四五六七八九十百千]+、)", text_no_title, flags=re.DOTALL)
if match:
    main_clause = match.group(1).strip()
else:
    # 如果沒有找到條列符號，則將整段視為主文        
    main_clause = text_no_title.strip()

# -------- 抓條文 --------
clauses = re.findall(r"([一二三四五六七八九十百千]+、)(.*?)(?=^([一二三四五六七八九十百千]+、)|$)", text_no_title, flags=re.DOTALL| re.MULTILINE)
clauses_dict = {clause[0].strip(): clause[1].strip() for clause in clauses}
if not clauses_dict:
    clauses_dict = None

# -------- 補充段：在最後一條條文之後 --------
if clauses:
    last_clause = clauses[-1][1]
    # 找出該段在原始內容中的結束位置
    last_clause_end = text_no_title.rfind(last_clause) + len(last_clause)
    remaining_text = text_no_title[last_clause_end:].strip()
    supplement_clause = remaining_text if remaining_text else None
else:
    supplement_clause = None

df_chunks = pd.DataFrame([{
    "chunk_id": f"{article_number}-{i+3}",
    "parent_id": article_number,
    "parent_title": article_title,
    "section": sec.strip("、"),
    "content": content.strip()
} for i, (sec, content) in enumerate(clauses_dict.items())
])

data_rows = [
    {
        "chunk_id": f"{article_number}-1",
        "parent_id": article_number,
        "parent_title": article_title,
        "section": "主文",
        "content": main_clause.strip()
    },
    {
        "chunk_id": f"{article_number}-2",
        "parent_id": article_number,
        "parent_title": article_title,
        "section": "補充項",
        "content": supplement_clause.strip()
    }
]
df_chunks = pd.concat([pd.DataFrame(data_rows), df_chunks], ignore_index=True)

print(df_chunks)