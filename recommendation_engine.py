import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

def model_init():
    with open("data/tfidf_vectorizer.pkl", "rb") as x:
        tfidf_vectorizer = pickle.load(x)

    return tfidf_vectorizer

def read_data():
    research_papers = pd.read_csv("data/research_papers.csv")

    with open("data/tfidf_matrix.pkl", "rb") as w:
        paper_vectors = pickle.load(w)

    return research_papers, paper_vectors

def preprocess_text(text: str) -> str:
    text = text.lower()  
    tokens = word_tokenize(text) 
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    
    return " ".join(tokens)

def classify_query_intent(query: str) -> str:
    # Convert query to lowercase
    query = query.lower()
    
    # Define keywords for papers and professors
    paper_keywords = ["paper", "research", "study", "article", "journal", "find", "recommend"]
    professor_keywords = ["professor", "collaborate", "researcher", "faculty", "expert", "mentor"]
    
    # Check if any keywords for papers or professors appear in the query
    if any(keyword in query for keyword in paper_keywords):
        return "papers"
    elif any(keyword in query for keyword in professor_keywords):
        return "professors"
    else:
        return "unknown"

def simplify_query(query: str, summarizer) -> str:
    # Simplify the query using the LLM
    simplified_query = summarizer(query, max_length=35, min_length=5, do_sample=False)[0]['summary_text']
    return simplified_query

def get_query_based_recommendations(query: str, tfidf_matrix, tfidf_vectorizer, research_papers, summarizer, num_rec: int) -> list:
    # Preprocess the query (similar preprocessing as your papers' abstracts)
    query = preprocess_text(query)
    
    # Vectorize the user query
    query_vector = tfidf_vectorizer.transform([query])  # Using the same tfidf_vectorizer for query vectorization
    
    # Calculate cosine similarity between query vector and all paper abstract vectors
    sim = cosine_similarity(query_vector, tfidf_matrix)  # shape (1, n_papers)
    sim = sim.reshape(sim.shape[1])  # Flatten to shape (n_papers,)
    
    # Get the indices of the top n similar papers (highest cosine similarity)
    top_n_idx = np.argsort(-sim)[:num_rec]
    recommended_papers = research_papers.iloc[top_n_idx].copy()

    # local_model_path = './local_bart_model'
    # summarizer = pipeline('summarization', model=local_model_path, max_length=100, device = 0)
    # summarizer = pipeline('summarization', model=local_model_path, max_length=100, device = 0)

    recommended_papers['Match Score'] = sim[top_n_idx]
    recommended_papers['combined_text'] = recommended_papers['title'] + " " + recommended_papers['abstract']

    # Running summarization
    summaries = [summarizer(paper['combined_text'])[0]['summary_text'] for _, paper in recommended_papers.iterrows()]
    recommended_papers.loc[:, 'Summary'] = summaries

    return recommended_papers[['title','abstract', 'authors', 'Summary', 'Match Score']]

def process_user_query(query: str, tfidf_matrix, tfidf_vectorizer, research_papers, summarizer, num_rec: int):
    intent = classify_query_intent(query)

    if intent == "unknown":
        response = ("It seems like your query is a bit unclear. Could you please specify:\n"
        "1. Are you looking for research papers in a specific topic?\n"
        "2. Are you interested in collaborating with professors or researchers?\n"
        "3. Or, are you seeking something else?\n"
        "Feel free to refine your query, and I can assist you better!")
        return response
        
    elif intent == "papers":
        # If the intent is to find papers, simplify the query and show paper recommendations
        simplified_query = simplify_query(query, summarizer)  # Use the previous simplification logic
        return get_query_based_recommendations(simplified_query, tfidf_matrix, tfidf_vectorizer, research_papers, summarizer, num_rec)
    
    elif intent == "professors":
        # If the intent is to find professors, match the query with professors
        simplified_query = simplify_query(query, summarizer)
        df = get_query_based_recommendations(simplified_query, tfidf_matrix, tfidf_vectorizer, research_papers, summarizer, num_rec)
        return df[['authors']]

if __name__ == "__main__":
    user_input = "quantum computing and machine learning"
    paper_vectors = ""

    tfidf_vectorizer = model_init()
    research_papers, paper_vectors = read_data()
    df = process_user_query(user_input, paper_vectors, tfidf_vectorizer, research_papers, 5)

    print(df)