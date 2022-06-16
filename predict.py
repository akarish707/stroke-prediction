from time import time
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

from imblearn.over_sampling import SMOTE

import joblib
import pickle

import time

def predict():
    st.header("Predict stroke")
    # st.subheader("Predicting stroke with KNN algorithm")
    st.write("Please insert your data by filling the form below.")


    def load_model():
        with open("model_knn.pkl", "rb") as file:
            data = pickle.load(file)
        sc = joblib.load("scaler.pkl")
        return data, sc
    
    def user_input():
        pass


    data, sc = load_model()

    gender_list = (
        "Male",
        "Female",
        "Other"
    )

    work_type_list = (
        "Private",
        "Self-employed",
        "children",
        "Govt_job",
        "Never_worked"
    )
    
    Residence_type_list = (
        "Urban",
        "Rural"
    )

    smoking_status_list = (
        "never smoked",
        "Unknown",
        "formerly smoked",
        "smokes"
    )
    
    # columns
    col1, col2 = st.columns(2)
    # left column
    with col1:

        gender = st.selectbox("Choose your gender", gender_list)
        hypertension = st.selectbox("Do you have Hypertension?", options=("Yes", "No"))
        heart_disease = st.selectbox("Do you have heart disease?", options=("Yes", "No"))
        ever_married = 	st.selectbox("Are you married?", options=("Yes", "No"))
        work_type = st.selectbox("Select your work type", options=work_type_list)
    
    # right column
    with col2:

        Residence_type = st.selectbox("Choose your residence type", options=Residence_type_list)
        smoking_status = st.selectbox("Do you smoke?", options=smoking_status_list)
        age = st.slider("Input your age", min_value=1, max_value=100, step=1)
        avg_glucose_level = st.slider("Average glucose level", min_value=50.0, max_value=300.0, step=0.1)
        bmi = st.slider("BMI", min_value=10.0, max_value=100.0, step = 0.1)
    
    submit = st.button("Predict")
    
    # if user click predict
    if submit:
        
        with st.spinner(text='In progress'):
            time.sleep(1)

        if hypertension == "Yes":
            hypertension = 1
        else:
            hypertension = 0

        if heart_disease == "Yes":
            heart_disease = 1
        else:
            heart_disease =0

        
        classifier_loaded = data["model"]
        le_gender = data["le_gender"]
        le_marriage = data["le_marriage"]
        le_work = data["le_work"]
        le_residence = data["le_residence"]
        le_smoke = data["le_smoke"]

        array = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])

        # st.write(array.transpose())

        array[:, 0] =  le_gender.transform(array[:, 0])
        array[:, 4] = le_marriage.transform(array[:, 4])
        array[:, 5] = le_work.transform(array[:, 5])
        array[:, 6] = le_residence.transform(array[:,6])
        array[:, 9] = le_smoke.transform(array[:,9])

        index_values = [0]
        column_values = ["gender", "age", "hypertension", "heart_disease","ever_married", "work_type", "Residence_type","avg_glucose_level", "bmi", "smoking_status"]

        X = pd.DataFrame(data = array, 
                        index = index_values,
                        columns = column_values)
        # st.write(X.transpose())
        X = sc.transform(X)
        y_pred = classifier_loaded.predict(X)
        st.write("----")
        st.header("Result")
        if y_pred == 0:
            st.subheader("You are classified to non-stroke category")
            st.image("asset/image/thumbs-up-doctor.jpg")
        else:
            st.subheader("You are classified to stroke category")
            st.image("asset/image/doctor-explains.jpg")
        
        # link_number = st.number_input("What subheading do you want to go to?", value=50)
        # st.markdown(f"<a href='#linkto_{link_number}'>Link to {link_number}</a>", unsafe_allow_html=True)

        # for i in range(100):
        #     st.markdown(f"<div id='linkto_{i}'></div>", unsafe_allow_html=True)
        #     st.subheader(f"Subtitle {i}")
        #     st.write(f"I am a thing {i}")