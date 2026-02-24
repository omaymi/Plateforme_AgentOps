import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

class VectorEngine:
    def __init__(self , documents):
        self.documents = documents
        self.tfidf_vectorizer = TfidfVectorizer()
        self.cbow_model = None

    def fit_tfidf(self):
        self.tfidf_vectorizer.fit_transform(self.documents)

    def fit_cbow(self):
        sentences=[doc.lower().split() for doc in self.documents]
        self.cbow_model = Word2Vec(sentences , vector_size=100 , window=5 , min_count=1 , sg=0)
    
    def get_cbow_average_vector(self, text):
        words = text.lower().split()
        vectors = [self.cbow_model.wv[word] for word in words if word in self.cbow_model.wv]
        if not vectors:
            return np.zeros(100)
        return np.mean(vectors, axis=0)

#moteur de recherche

    def search(self, query, method="tfidf"):
        if method == "tfidf":
            query_vec = self.tfidf_vectorizer.transform([query])
            doc_vecs = self.tfidf_vectorizer.transform(self.documents)
            similarities = cosine_similarity(query_vec, doc_vecs)
        
        elif method == "cbow":
            query_vec = self.get_cbow_average_vector(query).reshape(1, -1)
            doc_vecs = np.array([self.get_cbow_average_vector(doc) for doc in self.documents])
            similarities = cosine_similarity(query_vec, doc_vecs)
            
        index_max = np.argmax(similarities)
        return self.documents[index_max]



# # --- TEST RAPIDE ---
# if __name__ == "__main__":
#     docs = [
#         "Le chat mange de la souris.",
#         "Le langage Python est super pour l'IA.",
#         "Mistral est une entreprise française de technologie.",
#         "L'intelligence artificielle transforme le monde."
#     ]
    
#     engine = VectorEngine(docs)
#     engine.fit_tfidf()
#     engine.fit_cbow()
    
#     question = "est ce que les nouvelles technologies sont ameliorer la vie ?"

#     resultat = engine.search(question, method="tfidf")
#     print(f"Question: {question}\nDocument trouvé: {resultat}")