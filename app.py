import streamlit as st
from eda import *
from other import *
from predict import predict

def main():
    st.title("Stroke Prediction")
    st.write("----")

    page = st.selectbox("",['Home', 'Predict your health','Exploratory Data Analysis', 'About us'])

    if page =="Home":
        about_stroke()
    elif page =="Predict your health":
        predict()
    elif page == "Exploratory Data Analysis":
        pass
    elif page == "About us":
        about_us()

if __name__ == '__main__':
    main()