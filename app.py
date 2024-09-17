import streamlit as st
from predict_page import prediction
from explore_page import show_explore_page
from streamlit_navigation_bar import st_navbar

page = st_navbar(["Predict", "Explore"])

if page == 'Predict':
    prediction()
else:
    show_explore_page()
