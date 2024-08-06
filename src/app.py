from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
import torch.nn.functional as F
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key="up_bsQetBVyFTYszX7zw1tOHrU0RC9bm",  
    base_url="https://api.upstage.ai/v1/solar"
)

# 데이터 로딩
df = pd.read_excel('./sampled.xlsx')
df['DATA'] = df['DATA'].fillna('')
filtered_df = df[df['TARGET'] == 1]
texts = filtered_df['DATA'].tolist()
menu_names = filtered_df['MENU_NAME'].tolist()

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

embeddings = get_embeddings(texts)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    query_embedding = get_embeddings([user_input])[0]
    query_embedding = torch.tensor(query_embedding)

    embeddings_tensor = torch.tensor(embeddings)
    cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings_tensor, dim=1)

    top_k = 5
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for idx in top_results.indices:
        results.append({
            'menu': menu_names[idx].strip(),
            'text': texts[idx].strip(),
            'score': cos_scores[idx].item()
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
