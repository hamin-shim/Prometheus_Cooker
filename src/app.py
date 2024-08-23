from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
import torch.nn.functional as F
from flask_cors import CORS
import os
import pickle
from dotenv import load_dotenv
import os 
import re
from cal import calculate_bmr, calculate_tdee, calculate_calorie_deficit, calculate_diet_duration

# load .env
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL')
# PICKE_PATH = os.environ.get('PICKE_PATH')
PICKE_PATH = "../pickles"
print(OPENAI_API_KEY)
print(PICKE_PATH)


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
        # servings 정보 저장
        serving_info = value[4] if not (isinstance(value[4], float) and math.isnan(value[4])) else None
        servings.append(serving_info)
        
    return menu_names, np.array(embeddings), texts, nutrients, servings

def validate_input(user_input):
    # 정규 표현식 패턴
    pattern = r"^\d{2,3}/\d{2,3}/(남|여)/\d{1,2}/\d{2,3}$"
    
    # 입력값이 패턴과 일치하는지 확인
    if re.match(pattern, user_input):
        return True
    else:
        return False
    
def validate_range(input_value):
    try:
        # 입력값을 정수로 변환 시도
        value = int(input_value)
        
        # 변환된 값이 1에서 5 사이인지 확인
        if 1 <= value <= 5:
            return True
        else:
            return False
    except ValueError:
        # 변환이 불가능한 경우 (예: 입력값이 숫자가 아님)
        return False
    
def find_duration(user_info):
    results = []
    height, weight, gender, age, target_weight, activity_level = user_info.split('/')
    bmr = calculate_bmr(int(weight), int(height), int(age), gender.strip())
    tdee = calculate_tdee(bmr, int(activity_level))

    weight_loss = int(weight) - int(target_weight)
    calorie_deficit = calculate_calorie_deficit(weight_loss)

    diet_duration_days = calculate_diet_duration(calorie_deficit, tdee)
    results.append(f"\n당신의 기초대사량은 {bmr:.2f} kcal/day 입니다.")
    results.append(f"당신의 활동대사량은 {tdee:.2f} kcal/day 입니다.")
    results.append(f"감량해야 하는 칼로리는 총 {calorie_deficit:.2f} kcal 입니다.")
    results.append(f"예상 다이어트 기간은 약 {int(diet_duration_days)} 일입니다.")
    return results

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
            'score': cos_scores[idx].item(),
            'message': message  # 생성된 메시지를 추가
        })  
    return results


app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print(data)
    if 'text' not in data or 'category' not in data or 'type' not in data:
        return jsonify({'error': 'Missing parameters'}), 400

    input_text = data['text']
    category = data['category']
    if category == "": category = '쌀의 맛'
    data_type = data['type']

    if data_type == "recommendation":
        file_path = os.path.join(PICKE_PATH, f"{category}.pkl")
    # elif data_type == "calorie":
    #     file_path = os.path.join(EXCEL_PATH, f"{category}.xlsx")
    elif data_type == "weight_control":
        if(validate_input(input_text)):
            return jsonify({'type': 'weight', 'result': "pass", 'text':input_text})
        else: 
            return jsonify({'type': 'weight', 'result': "wrong"})
    elif data_type=='weight_control_activation':
        if(validate_range(input_text)):
            res = find_duration(f"{data['userInfo']}/{input_text}")
            return jsonify({'type': 'weight_finish', 'message': res})
        else: 
            return jsonify({'type': 'weight_2', 'result': "wrong"})
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
