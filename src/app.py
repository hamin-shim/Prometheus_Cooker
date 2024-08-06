from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
import torch.nn.functional as F
from flask_cors import CORS
import os
import pickle
import os
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
import torch.nn.functional as F
import pickle

PICKE_PATH = "/Users/hamin/Downloads/pickle"
OPENAI_API_KEY = "up_bsQetBVyFTYszX7zw1tOHrU0RC9bm"
OPENAI_BASE_URL = "https://api.upstage.ai/v1/solar"
cook_type_list = [i.split(".")[0] for i in os.listdir(PICKE_PATH)]
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
cook_type = cook_type_list[0]
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
def recommend(query):
    res = []

    # 코사인 유사도를 기준으로 입력 문장과 가장 유사한 5개 음식 추출
    top_k = 5
    query_embedding = get_embeddings([query])[0]
    query_embedding = torch.tensor(query_embedding)

    # 코사인 유사도 계산
    embeddings_tensor = torch.tensor(embeddings_np_array)
    cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings_tensor, dim=1)

    # 상위 top_k 결과 추출
    top_results = torch.topk(cos_scores, k=top_k)

    # print("\n\n======================\n\n")
    # print("사용자 입력 :", query)
    # print("\nTop 5 음식 추천:")

    # for idx in top_results.indices:
    #     print(menu_names[idx].strip()," : ",texts[idx].strip(),  "(Score: %.4f)" % (cos_scores[idx].item()))
    #     res.append(menu_names[idx].strip())
    results = []
    for idx in top_results.indices:
        results.append({
            'menu': menu_names[idx].strip(),
            'text': texts[idx].strip(),
            'score': cos_scores[idx].item()
        })
    print(results)
    return jsonify(results)

app = Flask(__name__)
CORS(app)
OPENAI_API_KEY = "up_bsQetBVyFTYszX7zw1tOHrU0RC9bm"

# OpenAI API 설정
openai_api_key = OPENAI_API_KEY
openai_api = OpenAI(api_key=openai_api_key)

# 간단한 텍스트 예측 모델 함수 (예시)
def predict_with_openai_model(input_text):
    response = openai_api.Completion.create(
        model="text-davinci-003",
        prompt=input_text,
        max_tokens=50
    )
    return response.choices[0].text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print(data)
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = data['text']
    res = recommend(input_text)
    print(res)
    return res

# 서버 상태 체크 엔드포인트
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'running'})

@app.route('/', methods=['GET'])
def hi():
    return jsonify({'status': 'Good'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
