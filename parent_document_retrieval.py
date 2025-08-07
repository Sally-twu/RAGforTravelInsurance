#import onnxruntime
#print(onnxruntime.__version__)

import re
import pandas as pd

# 範例法條內容
text = """
第七十二條 行動電話被竊損失保險理賠文件 
被保險人向本公司申請理賠時，應檢具下列文件：
一、理賠申請書。
二、當地警察機關刑事報案證明。如有必要時，本公司得要求被保險人提供行動電話相關購買
證明文件或證據。
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

# -------- 顯示結果 --------
print("條號:", article_number)
print("標題:", article_title)
print("主文:", main_clause)
if clauses_dict is not None:
    print("條文:")
    for k, v in clauses_dict.items():
        print(f"{k} {v}")
else:
    print("條文:", clauses_dict)
print("補充項:", supplement_clause)