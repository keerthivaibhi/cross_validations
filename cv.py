import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

st.title("Logistic Regression – Cross Validation")

data = load_breast_cancer()
X = data.data
y = data.target

kf = KFold(n_splits=3, shuffle=True, random_state=42)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

scores_logistic = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LogisticRegression(solver='liblinear', max_iter=1000)
    score = get_score(model, X_train, X_test, y_train, y_test)
    scores_logistic.append(score)

st.subheader("Cross Validation Scores")
st.write(scores_logistic)

st.subheader("Average Accuracy")
st.write(np.mean(scores_logistic))
