from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
import torch.nn.functional as F
from flask_cors import CORS
import os
import pickle

PICKE_PATH = "/Users/gamjawon/Desktop/Prometheus_Cooker/pickle"
EXCEL_PATH = "/Users/gamjawon/Desktop/Prometheus_Cooker/excel"
OPENAI_API_KEY = "up_bsQetBVyFTYszX7zw1tOHrU0RC9bm"
OPENAI_BASE_URL = "https://api.upstage.ai/v1/solar"
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

def load_dataset(file_path):
    if file_path.endswith(".pkl"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path).to_dict(orient='list')
    else:
        raise ValueError("Unsupported file format")
    return data

def prepare_data_for_recommendation(data):
    menu_names = []
    embeddings = []
    texts = []

    for key, value in data.items():
        menu_names.append(key)
        try:
            # Ensure that embeddings are numeric and can be converted to a numpy array
            embedding = np.array(value[0], dtype=np.float32)
        except ValueError:
            print(f"Problematic data for {key}: {value[0]}")
            raise ValueError(f"Embedding for {key} contains non-numeric data.")
        embeddings.append(embedding)
        texts.append(value[1])

    return menu_names, np.array(embeddings), texts

def recommend(query, data, data_type):
    res = []

    if data_type == 'recommendation':
        # 데이터가 이미 임베딩된 벡터로 되어 있는 경우
        menu_names, embeddings_np_array, texts = prepare_data_for_recommendation(data)
    elif data_type == 'calorie':
        # 텍스트 데이터를 기반으로 임베딩 벡터 생성
        menu_names = data['CKG_NM']
        texts = data['CKG_IPDC']
        embeddings_np_array = np.array(get_embeddings(texts))  # 텍스트를 임베딩 벡터로 변환
    else:
        raise ValueError("Unsupported data type")

    top_k = 5
    query_embedding = get_embeddings([query])[0]
    query_embedding = torch.tensor(query_embedding)

    embeddings_tensor = torch.tensor(embeddings_np_array, dtype=torch.float32)
    cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings_tensor, dim=1)

    top_results = torch.topk(cos_scores, k=top_k)

    results = [] 
    for idx in top_results.indices:
        results.append({
            'menu': menu_names[idx].strip(),
            'text': texts[idx].strip(),
            'score': cos_scores[idx].item()
        })  
    return jsonify(results)


app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if 'text' not in data or 'category' not in data or 'type' not in data:
        return jsonify({'error': 'Missing parameters'}), 400

    input_text = data['text']
    category = data['category']
    data_type = data['type']

    if data_type == "recommendation":
        file_path = os.path.join(PICKE_PATH, f"{category}.pkl")
    elif data_type == "calorie":
        file_path = os.path.join(EXCEL_PATH, f"{category}.xlsx")
    else:
        return jsonify({'error': 'Invalid data type'}), 400

    dataset = load_dataset(file_path)
    res = recommend(input_text, dataset, data_type)
    print(res)
    return res

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'running'})

@app.route('/', methods=['GET'])
def hi():
    return jsonify({'status': 'Good'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
