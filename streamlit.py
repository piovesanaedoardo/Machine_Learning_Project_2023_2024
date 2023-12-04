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

# sidebar
st.sidebar.header('Menù')

# title
st.title('Wine Quality Prediction')

# header
if st.sidebar.checkbox('DataFrame'):
    st.title('Dataframe')
    st.write(df)

# Description
if st.sidebar.checkbox('Description'):
    st.subheader('Dataframe Description')
    st.write(df.describe().T)

# Correlation Matrix
if st.sidebar.checkbox('Correlation Matrix'):
    st.subheader('Correlation Matrix')
    sns.heatmap(df.corr(), annot=True, fmt='.1f')
    st.pyplot(plt)

