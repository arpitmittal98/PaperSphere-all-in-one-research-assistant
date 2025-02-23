import pandas as pd
import streamlit as st
import plotly.express as px

st.logo(image="images/papersphere-logo-sidebar.png", icon_image='images/papersphere-logo-main.png', size="large")

st.set_page_config(initial_sidebar_state="collapsed")

st.header("Emerging Trends")

with st.sidebar:
    st.title("Welcome Arpit! :material/waving_hand:")
    st.write("\n")
    st.page_link("Home.py", label="Home", icon=":material/home:")
    st.page_link("pages/Explore.py", label="Explore", icon=":material/travel_explore:")
    st.page_link("pages/Trends.py", label="Trends", icon=":material/chart_data:")
    st.divider()
    st.page_link("pages/User.py", label="Your Account", icon=":material/person:")

@st.cache_data
def readDataset():
    df = pd.read_csv("data/research_papers_eda_dataset.csv")

    return df

df = readDataset()

option = st.selectbox(
    "Research Field",
    ("Computer Science", "Economics", "Electrical Engineering", "Mathematics", "Physics", "Quantitative Biology", "Quantitative Finance", "Statistics"),
    index=None
)

if option == "Computer Science":
    # Most researched sub-fields
    category_counts = df['definition'].value_counts()
    top_15_categories = category_counts.head(15)
    top_15_categories = pd.DataFrame(top_15_categories).reset_index()
    top_15_categories.columns = ["category", "count"]

    fig = px.pie(top_15_categories, values="count", names="category")

    st.subheader("Most researched sub-fields")
    st.plotly_chart(fig)

    df['update_date'] = pd.to_datetime(df['update_date'])
    df['year'] = df['update_date'].dt.year

    category_per_year_counts = df.groupby(['year', 'categories']).size().reset_index(name='count')
    most_researched_categories = category_per_year_counts.loc[category_per_year_counts.groupby('year')['count'].idxmax()]

    fig1 = px.bar(most_researched_categories, x="year", y="count", hover_data=["categories", "count"])
    st.subheader("Research Landscape")
    st.plotly_chart(fig1)

    filtered_df = df[df['journal-ref'] != 'Unavailable']

    popular_journal_refs_per_category = (
        filtered_df.groupby('categories')['journal-ref']
        .value_counts()
        .reset_index(name='count')
    )

    journalRef_per_category = popular_journal_refs_per_category.loc[
        popular_journal_refs_per_category.groupby('categories')['count'].idxmax()
    ]

    journalRef_per_category.columns = ['categories', 'most_referenced_journal_ref', 'count']

    fig2 = px.bar(journalRef_per_category, x="count", y="categories", hover_data=["most_referenced_journal_ref"], orientation='h', color_discrete_sequence=['teal'])
    fig2.update_layout(
        yaxis={'categoryorder':'total ascending'}
    )

    st.subheader("Most Published Journals")
    st.plotly_chart(fig2)