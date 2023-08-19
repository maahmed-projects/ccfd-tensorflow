import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
from tensorflow.keras.models import load_model
import pickle
tf_model = load_model('tf_model.json')
rfc_model = pickle.load(open('RandomForestClassifier_model.json','rb'))

st.header("Credit Card Fraud Detection")

st.subheader("Enter an input array")
input = st.text_input("Enter your input array: ", key="name")


if st.button('Make Prediction'):
    temp_array = input.split(',')
    temp_array = np.array(temp_array, dtype=np.float32)
    tf_pred_array = tf_model.predict(np.array( [temp_array]))
    rfc_pred_array = rfc_model.predict(np.array([temp_array]))
    tf_result = "Tensorflow Model predicts: "
    rfc_result = "Random Forest Classifier Model predicted: "
    if tf_pred_array[0] < .5:
        tf_result += "Action IS NOT fradualent"
    else:
        tf_result += "Action IS fradualent"
    if rfc_pred_array[0] < .5:
        rfc_result += "Action IS NOT fradualent"
    else:
        rfc_result += "Action IS fradualent"
    st.write(tf_result)
    st.write(rfc_result)
    
