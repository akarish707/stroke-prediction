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
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

import joblib
import pickle

def evaluation():
    # st.header("Data Preprocessing")
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")

    df = df.drop("id", axis=1)
    # st.write(df.head())
    # st.table(df.head())


    # fill missing bmi value with linear regression of age and gender
    X_bmi = df[['age','gender',"avg_glucose_level", 'bmi']].copy()

    le_gender = LabelEncoder()
    X_bmi["gender"] =  le_gender.fit_transform(X_bmi["gender"])

    Missing = X_bmi[X_bmi["bmi"].isna()]
    X_bmi = X_bmi[~X_bmi.bmi.isna()]
    Y_bmi = X_bmi.pop('bmi')

    linear_reg = LinearRegression()
    linear_reg.fit(X_bmi,Y_bmi)

    predicted_bmi = pd.Series(linear_reg.predict(Missing[['age','gender',"avg_glucose_level"]]),index=Missing.index)
    df.loc[Missing.index,'bmi'] = predicted_bmi
    df["bmi"] = df['bmi'].round(decimals = 1)
    
    le_gender = LabelEncoder()
    le_marriage = LabelEncoder()
    le_work = LabelEncoder()
    le_residence = LabelEncoder()
    le_smoke = LabelEncoder()

    df["gender"] =  le_gender.fit_transform(df["gender"])
    df["ever_married"] = le_marriage.fit_transform(df["ever_married"])
    df["work_type"] = le_work.fit_transform(df["work_type"])
    df["Residence_type"] = le_residence.fit_transform(df["Residence_type"])
    df["smoking_status"] = le_smoke.fit_transform(df["smoking_status"])

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    # st.write("Independent variables")
    # st.write(X)
    # st.write("Dependent variable")
    # st.write(y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

    from sklearn.neighbors import KNeighborsClassifier  
    best_k = -1
    best_test_score = -1

    for i in range(1,11):    
        classifier= KNeighborsClassifier(n_neighbors=i, p=1, weights = "uniform")  
    #     classifier.fit(X_train, y_train)
        classifier.fit(X_train_res, y_train_res.ravel())
        
        y_pred= classifier.predict(X_test)
        
        # st.write("**Neighbour** = {k}".format(k = i))
    #     print("The training accuracy: {score}".format(score = (classifier.score(X_train,y_train)*100).round(2)))
        # st.write("The training accuracy: {score}".format(score = (classifier.score(X_train_res,y_train_res.ravel())*100).round(2)))
        # st.write('The Test accuracy: {score}'.format(score = (classifier.score(X_test,y_test)*100).round(2)))
        # st.write("\n")
        if classifier.score(X_test,y_test) > best_test_score:
            best_test_score = classifier.score(X_test, y_test)
            best_k = i

    classifier= KNeighborsClassifier(n_neighbors=best_k, p=2)  
    classifier.fit(X_train_res, y_train_res.ravel())
    y_pred= classifier.predict(X_test)

    # evaluation starts here
    cnf_matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    plt.figure(figsize=(15,15))
    sns.heatmap(cnf_matrix, annot=True, fmt='g', ax=ax, cmap="YlGnBu")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Not Stroke(0)', 'Stroke(1)'])
    ax.yaxis.set_ticklabels(['Not Stroke(0)', 'Stroke(1)'])

    st.pyplot(fig)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    st.write("### Classification Report")
    st.text("\t" + classification_report(y_test, y_pred))
    st.write('Accuracy Score: ',accuracy_score(y_test,y_pred))
    








    