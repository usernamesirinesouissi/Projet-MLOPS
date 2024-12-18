import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Custom styling
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f4f8;  /* Light gray background */
        }
        .sidebar {
            background-color: #2D3A3C;  /* Dark Sidebar */
        }
        .sidebar .sidebar-content {
            color: white;  /* White text in sidebar */
        }
        .sidebar header {
            font-size: 20px;
            color: #ffffff;
        }
        .sidebar .sidebar-content button:hover {
            background-color: #ffcccb;
        }
        body {
            font-family: 'Helvetica', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Header and Title
st.title('Diabetes Checkup Prediction')
st.sidebar.header('Patient Data')

# Load Data
df = pd.read_csv("C:/Users/pc/Desktop/Projet ML/diabetes.csv")

# Load pre-trained model
loaded_model = pickle.load(open('C:/Users/pc/Desktop/Projet ML/diabet_model.sav', 'rb'))

# X and Y Data
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Sidebar Input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    return pd.DataFrame(user_report_data, index=[0])

# Patient Data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Predict with Model
user_result = loaded_model.predict(user_data)

# Display Prediction Results
st.subheader('Prediction Result')
if user_result[0] == 0:
    st.write('You are not Diabetic')
else:
    st.write('You are Diabetic')

# Visualizations
st.header('Visualized Patient Report')
fig_preg = plt.figure()
sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Set2')
sns.scatterplot(x=user_data['age'], y=user_data['pregnancies'], s=150, color='blue')
plt.title('Pregnancy Count vs Age')
st.pyplot(fig_preg)
