import pandas as pd
import numpy as np
import streamlit as st

st.title("Loan Approval predicition")
data = pd.read_csv("LoanApprovalPrediction.csv")


obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))

# Dropping Loan_ID column
data.drop(['Loan_ID'],axis=1,inplace=True)


# Import label encoder
from sklearn import preprocessing
	
# label_encoder object knows how
# to understand word labels.
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])


# To find the number of columns with
# datatype==object
obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))


for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
	
data.isna().sum()


from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.4,random_state=1)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators = 7,
							criterion = 'entropy',
							random_state =7)
svc = SVC()
lc = LogisticRegression()

# making predictions on the training set
for clf in (rfc, knn, svc,lc):
	clf.fit(X_train, Y_train)
	Y_pred = clf.predict(X_train)
	print("Accuracy score of ",
		clf.__class__.__name__,
		"=",100*metrics.accuracy_score(Y_train,
										Y_pred))


# making predictions on the testing set
for clf in (rfc, knn, svc,lc):
	clf.fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)
	print("Accuracy score of ",
		clf.__class__.__name__,"=",
		100*metrics.accuracy_score(Y_test,
									Y_pred))
        

a=st.text_input('Gender')
b=st.text_input('Married')
c=st.text_input('Dependent')
d=st.text_input('Education')
e=st.text_input('Self_Employed')
f=st.text_input('ApplicantIncome')
g=st.text_input('CoapplicantIncome')
h=st.text_input('LoanAmount')
i=st.text_input('Loan_Amount_Term')
j=st.text_input('Credit_History')
k=st.text_input('Property_Area')


if st.button('predict'):
    data=[[float(a),float(b),float(c),float(d),float(e),float(f),float(g),float(h),float(i),float(j),float(k)]]
    res=rfc.predict(data)
    st.success(res)