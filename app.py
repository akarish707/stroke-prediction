import streamlit as st
from eda import *
from other import *
from predict import *
from evaluation import *

def main():
    st.set_page_config(page_title="Stroke Prediction", page_icon="🩺")
    st.title("Stroke Prediction")
    st.write("----")

    page = st.selectbox("Navigation",['Home', 'Predict your health','Exploratory Data Analysis',"Evaluation Metric", 'About us'])

    if page =="Home":
        about_stroke()
    elif page =="Predict your health":
        predict()
    elif page == "Exploratory Data Analysis":
        eda()
    elif page == "Evaluation Metric":
        evaluation()
    elif page == "About us":
        about_us()

if __name__ == '__main__':
    main()