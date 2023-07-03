import streamlit as st
st.title("Iris Flower Classification")
sepal_length = st.text_input('Sepal Length')
sepal_width = st.text_input('Sepal Width')
petal_length = st.text_input('Petal Length')
petal_width = st.text_input('Petal Width')

import pandas as pd
import numpy as np
iris = pd.read_csv("IRIS.csv")

x = iris.drop("species", axis=1)
y = iris["species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)


	
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))

if(st.button('Predict')):
	Data = [[float(sepal_length),float(sepal_width),float(petal_length),float(petal_width)]]
	res=knn.predict(Data)
	st.success(res)