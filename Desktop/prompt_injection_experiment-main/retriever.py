import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K, NOISE_K, MIN_RELEVANCE_SCORE

vectorizer = None
doc_matrix = None
documents = None


def build_index(docs):

    global vectorizer, doc_matrix, documents

    documents = docs
    corpus = [d["text"] for d in docs]

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(corpus)


def retrieve(query: str):
    """
    Returns the top-K documents by TF-IDF cosine similarity, but only those
    scoring at or above MIN_RELEVANCE_SCORE. Previously this always returned
    the top-K regardless of how low the actual similarity was, meaning a
    completely out-of-scope query would still get "the least bad" documents
    from the corpus dressed up as relevant results. Now an out-of-scope
    query can return an empty list instead of forcing through low-confidence
    matches.
    """

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, doc_matrix)[0]

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    confident = [
        {"doc": d, "retrieval_score": float(s)}
        for d, s in ranked[:TOP_K]
        if s >= MIN_RELEVANCE_SCORE
    ]

    if not confident:
        print(f"[retriever] No confident match for query (all scores below {MIN_RELEVANCE_SCORE}): {query!r}")

    return confident


def retrieve_with_noise(query: str):

    top = retrieve(query)
    top_docs = [x["doc"] for x in top]

    remaining = [d for d in documents if d not in top_docs]
    noise = random.sample(remaining, min(NOISE_K, len(remaining)))

    noise_items = [
        {"doc": d, "retrieval_score": 0.0}
        for d in noise
    ]

    return top + noise_items