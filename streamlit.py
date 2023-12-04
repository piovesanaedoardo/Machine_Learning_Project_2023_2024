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
# st.title('Wine Quality Prediction')

# header
st.header('Data Information:')
st.write(df.describe())

# show data as table
st.header('Data:')
st.write(df)