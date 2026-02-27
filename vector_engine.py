import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

class VectorEngine:
    def __init__(self, documents):
        self.documents = documents
        self.tfidf_vectorizer = None
        self.tfidf_doc_vectors = None
        self.cbow_model = None
        self.cbow_doc_vectors = None

    # ---------- TF-IDF ----------
    def fit_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_doc_vectors = self.tfidf_vectorizer.fit_transform(self.documents)

    # ---------- CBOW ----------
    def fit_cbow(self):
        sentences = [doc.lower().split() for doc in self.documents]
        self.cbow_model = Word2Vec(
            sentences,
            vector_size=100,
            window=5,
            min_count=1,
            sg=0
        )

        self.cbow_doc_vectors = np.array(
            [self.get_cbow_average_vector(doc) for doc in self.documents]
        )

    def get_cbow_average_vector(self, text):
        words = text.lower().split()
        vectors = [
            self.cbow_model.wv[word]
            for word in words
            if word in self.cbow_model.wv
        ]

        if not vectors:
            return np.zeros(self.cbow_model.vector_size)

        return np.mean(vectors, axis=0)

    # ---------- SEARCH ----------
    def search(self, query, method="tfidf", top_k=3):

        # Garde : le vectoriseur n'a pas encore été entraîné (aucun document indexé)
        if method == "tfidf" and (self.tfidf_vectorizer is None or not self.documents):
            return []
        if method == "cbow" and (self.cbow_model is None or self.cbow_doc_vectors is None):
            return []

        if method == "tfidf":
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_doc_vectors)

        elif method == "cbow":
            query_vec = self.get_cbow_average_vector(query).reshape(1, -1)
            similarities = cosine_similarity(query_vec, self.cbow_doc_vectors)

        else:
            raise ValueError("Méthode non supportée")

        # récupérer top-k
        top_indices = similarities[0].argsort()[-top_k:][::-1]

        return [self.documents[i] for i in top_indices]