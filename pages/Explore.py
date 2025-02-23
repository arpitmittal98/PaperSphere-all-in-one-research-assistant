import pickle
import pandas as pd
import streamlit as st
import recommendation_engine as rec

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

def display_recommendations(df: pd.DataFrame) -> None:
    if len(df.columns) > 1:
        for index, row in df.iterrows():
            with st.expander(f"{row['title']}"):
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.markdown("**Summary**")
                    st.markdown(f"*{row['Summary']}*")
                with col2:
                    st.metric(label="Match", value=f"{row['Match Score'] * 100:.1f}%", border=True)

                st.markdown("**Abstract**")
                st.markdown(row["abstract"])
    else:
        for index, row in df.iterrows():
            with st.expander(f"{row['authors']}"):
                st.markdown(f"{row['authors']}@buffalo.edu")
                st.markdown("University at Buffalo")
                st.markdown("School of Engineering and Applied Sciences")


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if (message["role"] == "user") or (type(message["content"]) == str):
            st.markdown(message["content"])
        else:
            display_recommendations(message["content"])

if query := st.chat_input("How can I help you today?"):
    st.session_state["messages"].append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Finding your best match...", show_time=True):
            out = rec.process_user_query(
                query, 
                st.session_state["paper_vectors"], 
                st.session_state["vectorizer"], 
                st.session_state["research_papers"], 
                st.session_state["summarizer"], 
                5)
            if type(out) == pd.DataFrame:
                st.write("Here are your best matches:")
                display_recommendations(out)
            else:
                st.write(out)
        
        st.session_state["messages"].append({"role": "assistant", "content": out})
        

