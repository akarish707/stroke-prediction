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

def eda():
    st.header("Data Preprocessing")
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")

    st.markdown("""
    
    ### Column Descriptions :
    - id: unique identifier
    - gender: "Male", "Female" or "Other"
    - age: age of the patient
    - hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
    - heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
    - ever_married: "No" or "Yes"
    - work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
    - Residence_type: "Rural" or "Urban"
    - avg_glucose_level: average glucose level in blood
    - bmi: body mass index
    - smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
    - stroke: 1 if the patient had a stroke or 0 if not

    Note: "Unknown" in smoking_status means that the information is unavailable for this patient

        """)

    st.write("This is how the raw data looks like")
    st.write(df.head())
    st.write("The shape of the data: ", df.shape)
    st.write("Since id column is not going to help us predict anything at all, we might as well just drop it from the DataFrame.")

    df = df.drop("id", axis=1)
    st.write(df.head())
    # st.table(df.head())

    st.write("Then we check for any empty data in our stroke data")
    st.write(df.isnull().sum())
    st.write("We see in the graph below that BMI has **201** empty data")
    
    
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(10, 10))
    plt.title('Missing Value Status')
    sns.heatmap(df.isna().sum().to_frame(),annot=True,fmt='d',cmap='YlGnBu')
    ax.set_xlabel('Amount Missing')
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    st.write("Since BMI varies between age, gender, and average glucose level, we didn't just change them based on average, median, or mode of values in BMI, therefore we decided to fill the empty value of BMI based on linear regression value between age, gender and average glucose level with target of BMI.")

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

    st.code("""
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
    
    """)
    st.write("Here is the result:")
    st.write(df.iloc[:100,:])

    st.write("---")
    st.header("Analysis of the whole dataset")
    st.write("Let's check unique values within our categorical features")

    columns = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status','stroke']
    for i in range(len(columns)):
        st.write("Unique value of column {column}:".format(column=columns[i]))
        st.write(df[columns[i]].value_counts(),"\n")
    
    st.write("Now let's check the number of cases from stroke and not stroke ")

    labels =df['stroke'].value_counts(sort = True).index
    sizes = df['stroke'].value_counts(sort = True)
    colors = ["lightblue","red"]
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(4,4))
    plt.title('Number of stroke in the dataset')
    plt.pie(sizes, explode=(0.05,0), labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,)
    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    st.warning("Based on the pie chart above, we can conclude that the data is imbalance, because there is less than 5% of data belong to stroke class, and the rest of more than 95% belong to non-stroke class. If we leave this as it is, we might get a high accuracy for non-stroke class, but very low accuracy for stroke class. There is another way rather than just leaving them imbalance, later down below we are going to use SMOTE technique to increase the number of low data from a class.")

    st.write("Here is the histogram of BMI, Age, and average glucose level in blood")

    def plot_hist(col, bins=30, title="",xlabel="",ax=None):
        sns.histplot(col, bins=bins,ax=ax)
        ax.set_title(f'Histogram of {title}',fontsize=15)
        ax.set_xlabel(xlabel)

    fig, axes = plt.subplots(1,3,figsize=(11,7),constrained_layout=True)
    plot_hist(df.bmi,
            title='Bmi',
            xlabel="Level of the BMI",
            ax=axes[0])
    plot_hist(df.age,
            bins=30,
            title='Age',
            xlabel='Age',
            ax=axes[1])
    plot_hist(df.avg_glucose_level,
            title='Serum Creatinine', 
            xlabel='Level of serum creatinine in the blood (mg/dL)',
            ax=axes[2])

    st.pyplot(fig)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    ax = sns.catplot(y="work_type", 
            hue="stroke", 
            kind="count",
            palette="pastel", 
            edgecolor=".6",
            data=df)
    st.pyplot(ax)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    ax = sns.catplot(y="smoking_status", 
            hue="stroke", 
            kind="count",
            palette="pastel", 
            edgecolor=".6",
            data=df)
    st.pyplot(ax)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


    fig, ax = plt.subplots()
    ax = sns.catplot(x="gender", y="stroke", hue="heart_disease", palette="pastel", kind="bar", data=df)
    st.pyplot(ax)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    ax = sns.catplot(x="gender", y="stroke", hue="Residence_type", palette="pastel", kind="bar", data=df)
    st.pyplot(ax)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    ax = sns.catplot(x="gender", y="stroke", hue="hypertension", palette="pastel", kind="bar", data=df)
    st.pyplot(ax)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    le = LabelEncoder()
    en_df = df.apply(le.fit_transform)
    en_df.head()
    
    st.write("### Heatmap Corellation")
    fig, ax = plt.subplots()
    plt.figure(figsize=(15,15))
    sns.heatmap(en_df.corr(), ax=ax, annot=True, fmt=".2f", cmap = 'YlGnBu')
    st.write(fig)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    
    fig, ax = plt.subplots()
    corr = en_df.corr()[['stroke']].sort_values(by='stroke', ascending=False)
    sns.heatmap(corr, vmin=-1, vmax=1, ax=ax, annot=True, cmap = 'BrBG')
    plt.figure(figsize=(8, 12))

    st.pyplot(fig)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    st.header("Machine Learning Modelling")
    st.subheader("Prediction using KNN")

    st.write("### Encode Categorical Variable")

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

    st.code("""
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

    """)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    st.write("Independent variables")
    st.write(X)
    st.write("Dependent variable")
    st.write(y)


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    st.write("### Test and training data splitting")
    st.code("""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    """)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    st.write("### Standard scaling")
    st.code("""
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    """)

    st.write("""
    
    ### SMOTE( Synthetic Minority Over-Sampling Technique)
    Used to handle imbalance data

    """)

    st.code("""
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
    """)

    st.write("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    st.write("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

    st.write('After OverSampling, the shape of X_train: {}'.format(X_train_res.shape))
    st.write('After OverSampling, the shape of y_train: {} \n'.format(y_train_res.shape))

    st.write("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    st.write("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

    st.write("### KNN")
    st.write("K-nearest neighbor")
    st.write("Finding the best K neighbor with highest test accuracy")

    st.code("""
    from sklearn.neighbors import KNeighborsClassifier  
    best_k = -1
    best_test_score = -1

    for i in range(1,11):    
        classifier= KNeighborsClassifier(n_neighbors=i, p=1, weights = "uniform")  
    #     classifier.fit(X_train, y_train)
        classifier.fit(X_train_res, y_train_res.ravel())
        
        y_pred= classifier.predict(X_test)
        
        print("Neighbour = {k}".format(k = i))
    #     print("The training accuracy: {score}".format(score = (classifier.score(X_train,y_train)*100).round(2)))
        print("The training accuracy: {score}".format(score = (classifier.score(X_train_res,y_train_res.ravel())*100).round(2)))
        print('The Test accuracy: {score}'.format(score = (classifier.score(X_test,y_test)*100).round(2)))
        
        if classifier.score(X_test,y_test) > best_test_score:
            best_test_score = classifier.score(X_test, y_test)
            best_k = i

    print("Best k-neighbor= {k}".format(k = best_k))
    """)

    from sklearn.neighbors import KNeighborsClassifier  
    best_k = -1
    best_test_score = -1

    for i in range(1,11):    
        classifier= KNeighborsClassifier(n_neighbors=i, p=1, weights = "uniform")  
    #     classifier.fit(X_train, y_train)
        classifier.fit(X_train_res, y_train_res.ravel())
        
        y_pred= classifier.predict(X_test)
        
        st.write("**Neighbour** = {k}".format(k = i))
    #     print("The training accuracy: {score}".format(score = (classifier.score(X_train,y_train)*100).round(2)))
        st.write("The training accuracy: {score}".format(score = (classifier.score(X_train_res,y_train_res.ravel())*100).round(2)))
        st.write('The Test accuracy: {score}'.format(score = (classifier.score(X_test,y_test)*100).round(2)))
        st.write("\n")
        if classifier.score(X_test,y_test) > best_test_score:
            best_test_score = classifier.score(X_test, y_test)
            best_k = i

    st.write("#### Best k-neighbor= {k}".format(k = best_k))

    st.write("### Predicting with KNN")

    classifier= KNeighborsClassifier(n_neighbors=best_k, p=2)  
    classifier.fit(X_train_res, y_train_res.ravel())
    y_pred= classifier.predict(X_test)

    st.code("""
    classifier= KNeighborsClassifier(n_neighbors=best_k, p=2)  
classifier.fit(X_train_res, y_train_res.ravel())
y_pred= classifier.predict(X_test)
    """)

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
    








    