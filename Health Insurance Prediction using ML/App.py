import streamlit as st
st.title("Health Insurance Premium Prediction")
Age= st.text_input('Age')
Sex = st.text_input('Sex')
BMI = st.text_input('BMI')
Smoker = st.text_input('Smoker')

import numpy as np
import pandas as pd
data = pd.read_csv("Health_insurance.csv")
data.head()

data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
print(data.head())

x = np.array(data[["age", "sex", "bmi", "smoker"]])
y = np.array(data["charges"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(xtrain, ytrain)

ypred = forest.predict(xtest)
data = pd.DataFrame(data={"Predicted Premium Amount": ypred})
print(data.head())

if(st.button('Predict')):
	Data = [[float(Age),float(Sex),float(BMI),float(Smoker)]]
	res=forest.predict(Data)
	st.success(res)