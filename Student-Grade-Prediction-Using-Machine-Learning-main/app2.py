import numpy as np
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

st.title("Students grade prediction")
data = pd.read_csv('student-mat.csv')

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop("G3", axis=1))
y = np.array(data[predict])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

linear_regression = LinearRegression()
linear_regression.fit(xtrain, ytrain)
accuracy = linear_regression.score(xtest, ytest)

st.write("Model Accuracy:", accuracy)

a = st.text_input('studytime')
b = st.text_input('failures')
c = st.text_input('absences')
d = st.text_input('G1')
e = st.text_input('G2')

if st.button('predict'):
    input_data = [[float(a), float(b), float(c), float(d), float(e)]]
    predictions = linear_regression.predict(input_data)
    st.success("Predicted Grade (G3): {}".format(predictions[0]))
