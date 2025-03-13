from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel('/home/parthieshwar/Development/Django/Backend/ResumeApp/Book.xlsx')

unique_desc_df = df.drop_duplicates(subset=['Description']).reset_index(drop=True)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
unique_embeddings = model.encode(unique_desc_df['Description'].tolist())

dbscan = DBSCAN(eps=0.5, min_samples=5)
unique_desc_df['cluster'] = dbscan.fit_predict(unique_embeddings)

df = df.merge(unique_desc_df[['Description', 'cluster']], on='Description', how='left')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
role_encoded = encoder.fit_transform(df[['Role']])

embeddings = model.encode(df['Description'].tolist())

def find_related_descriptions(user_input, data, model, embeddings, encoder, role_encoded):
    exact_matches = data[data['Role'].str.lower() == user_input.lower()]
    user_embedding = model.encode([user_input])

    role_input_encoded = encoder.transform([[user_input]])
    role_similarity = cosine_similarity(role_input_encoded, role_encoded).flatten()

    text_similarity = cosine_similarity(user_embedding, embeddings).flatten()
    overall_similarity = (role_similarity + text_similarity) / 2

    top_indices = np.argsort(overall_similarity)[::-1][:30]
    related_descriptions = data.iloc[top_indices][['Description', 'cluster']]

    results = pd.concat([exact_matches[['Description', 'cluster']], related_descriptions]).drop_duplicates(subset=['Description', 'cluster']).reset_index(drop=True)
    return results[['Description']].to_dict(orient='records')

def get_descriptions(request):
    if request.method == 'POST':
        user_input = request.POST.get('role')
        related_descriptions = find_related_descriptions(user_input, df, model, embeddings, encoder, role_encoded)
        return JsonResponse(related_descriptions, safe=False)
    return render(request, 'descriptions/index.html')