import os
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
import torch.nn.functional as F
import pickle

PICKE_PATH = "/Users/hamin/Downloads/pickle"
SAMPLE_EXCEL_PATH = "./sampled.xlsx"
OPENAI_API_KEY = "up_bsQetBVyFTYszX7zw1tOHrU0RC9bm"
OPENAI_BASE_URL = "https://api.upstage.ai/v1/solar"
cook_type_list = [i.split(".")[0] for i in os.listdir(PICKE_PATH)]


# OpenAI 클라이언트 설정
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)
def get_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch_texts,
            model="solar-embedding-1-large-query"
        )
        batch_embeddings = [np.array(embedding.embedding) for embedding in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

cook_type = cook_type_list[int(input({i:c for i,c in enumerate(cook_type_list)}))]

with open(os.path.join(PICKE_PATH, f"{cook_type}.pkl"), 'rb') as f:
    dic_read = pickle.load(f)

# 딕셔너리 전처리
menu_names = []
embeddings = []
texts = []

for key, value in dic_read.items():
    menu_names.append(key)
    embeddings.append(value[0])
    texts.append(value[1])

embeddings_np_array = np.array(embeddings)

# 사용자 입력 문장
queries = [input("what's your request?")]

# 코사인 유사도를 기준으로 입력 문장과 가장 유사한 5개 음식 추출
top_k = 5
for query in queries:
    query_embedding = get_embeddings([query])[0]
    query_embedding = torch.tensor(query_embedding)

    # 코사인 유사도 계산
    embeddings_tensor = torch.tensor(embeddings_np_array)
    cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings_tensor, dim=1)

    # 상위 top_k 결과 추출
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("사용자 입력 :", query)
    print("\nTop 5 음식 추천:")

    for idx in top_results.indices:
        print(menu_names[idx].strip()," : ",texts[idx].strip(),  "(Score: %.4f)" % (cos_scores[idx].item()))




