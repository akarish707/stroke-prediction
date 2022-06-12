import streamlit as st
from eda import *
from other import *
from predict import predict

def main():
    st.title("Stroke Prediction")
    st.write("----")

    page = st.selectbox("Navigation",['Home', 'Predict your health','Exploratory Data Analysis',"Evaluation Metric", "Result Visualization", 'About us'])

    if page =="Home":
        about_stroke()
    elif page =="Predict your health":
        predict()
    elif page == "Exploratory Data Analysis":
        eda()
    elif page == "About us":
        about_us()

if __name__ == '__main__':
    main()