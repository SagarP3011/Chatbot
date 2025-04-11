from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# === Input texts ===

ai_answer = """ """
chatbot_answer = """ """


# === 1. TF-IDF Cosine Similarity ===
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([ai_answer, chatbot_answer])
tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print(f"TF-IDF Cosine Similarity: {tfidf_sim:.4f}")

# === 2. Sentence Embedding Cosine Similarity using MPNet ===
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode([ai_answer, chatbot_answer], convert_to_tensor=True)
embedding_sim = cosine_similarity(
    embeddings[0].cpu().numpy().reshape(1, -1),
    embeddings[1].cpu().numpy().reshape(1, -1)
)[0][0]

print(f"MPNet Embedding Cosine Similarity: {embedding_sim:.4f}")
