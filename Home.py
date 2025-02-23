import pickle
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(initial_sidebar_state="collapsed")

st.logo(image="images/papersphere-logo-sidebar.png", icon_image='images/papersphere-logo-main.png', size="large")

with st.sidebar:
    st.title("Welcome Arpit! :material/waving_hand:")
    st.write("\n")
    st.page_link("Home.py", label="Home", icon=":material/home:")
    st.page_link("pages/Explore.py", label="Explore", icon=":material/travel_explore:")
    st.page_link("pages/Trends.py", label="Trends", icon=":material/chart_data:")
    st.divider()
    st.page_link("pages/User.py", label="Your Account", icon=":material/person:")

st.title("PaperSphere Says Hi!", )
col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.image("images/papersphere-logo-main.png")
    st.page_link("pages/Explore.py", label="Explore your interests", icon=":material/explore:", use_container_width=True)
with col2:
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.markdown("**PaperSphere** is a platform to explore your research interests to the fullest and connect you with the right mentors on your research journey.\
                Whether you are a novice or a seasoned academician, PaperSphere is ready to help you along the way. All you have to do is ask! It is as simple as that. ")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.page_link("pages/Trends.py", label="What is Trending", icon=":material/chart_data:", use_container_width=True)

if "research_papers" not in st.session_state:
    research_papers = pd.read_csv("data/research_papers.csv")
    st.session_state["research_papers"] = research_papers
if "paper_vectors" not in st.session_state:
    with open("data/tfidf_matrix.pkl", "rb") as w:
        paper_vectors = pickle.load(w)
    st.session_state["paper_vectors"] = paper_vectors
if "vectorizer" not in st.session_state:
    with open("data/tfidf_vectorizer.pkl", "rb") as x:
        tfidf_vectorizer = pickle.load(x)
    st.session_state["vectorizer"] = tfidf_vectorizer
if "summarizer" not in st.session_state:
    local_model_path = './local_bart_model'
    summarizer = pipeline('summarization', model=local_model_path, max_length=100, device = 0)
    st.session_state["summarizer"] = summarizer
if "messages" not in st.session_state:
    st.session_state["messages"] = []