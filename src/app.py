from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
import torch.nn.functional as F
from flask_cors import CORS
import os
import pickle
import math
from dotenv import load_dotenv

load_dotenv()

PICKE_PATH = os.getenv('PICKE_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
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
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_data_for_recommendation(data):
    menu_names = []
    embeddings = []
    texts = []
    nutrients = []
    servings = []  # servings (몇 인분인지) 저장할 리스트 추가

    for key, value in data.items():
        menu_names.append(key)
        try:
            embedding = np.array(value[0], dtype=np.float32)
        except ValueError:
            print(f"Problematic data for {key}: {value[0]}")
            raise ValueError(f"Embedding for {key} contains non-numeric data.")
        embeddings.append(embedding)
        texts.append(value[1])

        # nutrients 딕셔너리로 저장
        nutrient_info = value[3]
        print(f"Nutrient info for {key}: {nutrient_info}")  # 여기를 통해 데이터 구조를 확인
        if isinstance(nutrient_info, str):  # 만약 nutrient_info가 문자열이면
            try:
                nutrient_info = eval(nutrient_info)  # 문자열을 딕셔너리로 변환
            except:
                nutrient_info = {}  # 변환 실패 시 빈 딕셔너리로 대체
        nutrients.append(nutrient_info)

        serving_info = value[4] if not (isinstance(value[4], float) and math.isnan(value[4])) else None
        servings.append(serving_info)

    return menu_names, np.array(embeddings), texts, nutrients, servings


def recommend(query, data):
    menu_names, embeddings_np_array, texts, nutrients, servings = prepare_data_for_recommendation(data)

    top_k = 5
    query_embedding = get_embeddings([query])[0]
    query_embedding = torch.tensor(query_embedding)

    embeddings_tensor = torch.tensor(embeddings_np_array, dtype=torch.float32)
    cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings_tensor, dim=1)

    top_results = torch.topk(cos_scores, k=top_k)

    results = [] 
    for idx in top_results.indices:
        nutrient_info = nutrients[idx] if nutrients[idx] else {}
        serving_info = servings[idx]  # servings 정보 추가

        # 챗봇 응답 메시지에 serving_info가 있으면 추가
        message = f"{menu_names[idx].strip()}의 칼로리는 {nutrient_info.get('Cal', '알 수 없음')}kcal, 지방 {nutrient_info.get('지방', '알 수 없음')}, 탄수화물 {nutrient_info.get('탄수화물', '알 수 없음')}, 단백질 {nutrient_info.get('단백질', '알 수 없음')}이 들어있어요!"
        
        if serving_info is not None:
            message += f" **{serving_info} 기준**"

        results.append({
            'menu': menu_names[idx].strip(),
            'text': texts[idx].strip(),
            'nutrients': nutrient_info,
            'serving': serving_info,
            'message': message  # 생성된 메시지를 추가
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
        file_path = os.path.join(PICKE_PATH, f"{category}.pkl")
    else:
        return jsonify({'error': 'Invalid data type'}), 400

    dataset = load_dataset(file_path)
    res = recommend(input_text, dataset)
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
