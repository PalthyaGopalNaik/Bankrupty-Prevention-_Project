# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading The Saved Model.
loaded_model=pickle.load(open('D:/Project  Deployment/trained_model.sav','rb'))





input_data=(0.0,1.0,1.0,1.0,1.0,0.5)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
# Output the prediction
if prediction[0] == 1:
    print("The Company is bankruptcy")
else:
    print("The Company is non-bankruptcy")
