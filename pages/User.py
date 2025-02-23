import streamlit as st

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

st.title("User Profile")
st.divider()
# Create two columns for layout
col1, col3, col2 = st.columns([1, 0.3, 2])

with col1:
    # Profile picture placeholder
    st.image("images/dp-placeholder.jpg", caption="Arpit Mittal")

with col2:
    # Mock user information
    st.subheader("Arpit Mittal")
    st.write("📧 arpit@buffalo.edu")
    st.write("🏢 Innovation Labs")
    st.write("💼 Data Scientist")
    
    # Display interests as tags
    st.markdown("**Interests**: Machine Learning, Databases, Natural Language Processing, Python")