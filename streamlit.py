# streamlit app for the project
# to run: streamlit run streamlit.py

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv('winequality-red.csv')

# title
st.title('Wine Quality Prediction')

# header
st.header('Dataframe')
st.write(df)

# Description
st.subheader('Dataframe Description')
st.write(df.describe().T)

# Correlation Matrix
st.subheader('Correlation Matrix')
sns.heatmap(df.corr(), annot=True, fmt='.1f')
st.pyplot(plt)
