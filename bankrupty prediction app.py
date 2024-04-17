# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:40:31 2024

@author: palth
"""


import numpy as np
import pickle
import streamlit as st



# Loading The Saved Model.
loaded_model=pickle.load(open('D:/Project  Deployment/trained_model.sav','rb'))
 
 
# Creating A Fuction For Prediction
 
def bankrupty_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    # Output the prediction
    if prediction[0] == 1:
        return "The Company is bankruptcy"
    else:
        return "The Company is non-bankruptcy"
     
     
     
     
        
     
        
     
        
def main():
    
    
    # Giving Title
    st.title("Bankrupty Prediction System")
    
    
    # Getting Input From The User
    
    # Input features
    industrial_risk = st.selectbox('Industrial Risk', [0, 0.5, 1])
    management_risk = st.selectbox('Management Risk', [0, 0.5, 1])
    financial_flexibility = st.selectbox('Financial Flexibility', [0, 0.5, 1])
    credibility = st.selectbox('Credibility', [0, 0.5, 1])
    competitiveness = st.selectbox('Competitiveness', [0, 0.5, 1])
    operating_risk = st.selectbox('Operating Risk', [0, 0.5, 1])
    
    
    


    
    
    
    # Code For Prediction
    
    diagnosis=''
    
    # Creating A Button For Prediction
    
    if st.button('Prediction Result'):
        diagnosis=bankrupty_prediction([industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk])
        
        
    
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()