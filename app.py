import streamlit as st
from predict_page import prediction
from explore_page import show_explore_page

page = st.sidebar.selectbox(
    "Explore Or Predict", ("Predict", "Explore"))

if page == 'Predict':
    prediction()
else:
    show_explore_page()
