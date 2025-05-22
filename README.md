# PaperSphere - Research Paper Recommendaton using NLP and LLMs

**PaperSphere** is an AI-powered research assistant designed to help users discover and explore academic papers more efficiently. By leveraging large language models and semantic search, PaperSphere provides intelligent paper recommendations, semantic query understanding, and a personalized research experience.

## Features

- **Semantic Search**: Understands user queries and compares them with research paper abstracts using sentence embeddings.
- **LLM Integration**: Uses locally stored LLMs to summarize queries and assist in recommendation logic.
- **Paper and Professor Recommendations**: Matches queries with top-ranked research papers and relevant professors.
- **Talk to Document (Prototype)**: Enables querying specific documents (e.g., research papers) using natural language.
- **Streamlit-based UI**: Interactive front-end to simplify the research discovery process.

## Tech Stack

- **Python** for backend logic
- **Hugging Face Transformers** and **Sentence Transformers** for LLM and semantic search
- **Streamlit** for the web interface
- **Local JSON/CSV datasets** for papers and professor profiles

## Dataset
 We used arXiv open source dataset from Cornell University. This dataset is updated weekly and has around 1.7 million entries. <br>
 <br>
 Link for the dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv

