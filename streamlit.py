# streamlit app for the project
# streamlit run streamlit.py

# import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

# load data
df = pd.read_csv('winequality-red.csv')


# Function to plot histograms
def plot_histograms(df):
    for column in df.columns[:-1]:  # Excluding the target variable 'quality'
        st.subheader(f'Histogram for {column}')
        fig, ax = plt.subplots()
        df[column].hist(bins=20, ax=ax)
        st.pyplot(fig)

# Function to plot box plots
def plot_box_plots(df):
    for column in df.columns[:-1]:  # Excluding the target variable 'quality'
        st.subheader(f'Box Plot for {column}')
        fig, ax = plt.subplots()
        df.boxplot(column=column, ax=ax)
        st.pyplot(fig)

# sidebar
st.sidebar.header('Menù')

# title
st.title('Wine Quality Prediction App')

# Dataframe
if st.sidebar.checkbox('DataFrame'):
    st.title('Dataframe')
    st.write(df)

# Description
if st.sidebar.checkbox('Description'):
    st.subheader('Dataframe Description')
    st.write(df.describe().T)

    # Dataframe Shape
    st.subheader('Dataframe Shape')
    st.write(df.shape)

    # Columns types
    st.subheader('Columns types')
    st.write(df.dtypes.rename("Type"))

    # Target variable
    st.subheader('Target variable distribution')
    st.write(df['quality'].value_counts().sort_index(ascending=True))

    # Add options to display histograms and box plots
    if st.checkbox('Show Histograms of Features'):
        plot_histograms(df)

    if st.checkbox('Show Box Plots of Features'):
        plot_box_plots(df)

# Correlation Matrix
if st.sidebar.checkbox('Correlation Matrix'):
    st.subheader('Correlation Matrix')
    sns.heatmap(df.corr(), annot=True, fmt='.1f')
    st.pyplot(plt)

    st.write("The most correlated features are: \n")
    st.write(df.corr().unstack().sort_values(ascending=True).drop_duplicates().tail(6)[:-1].rename_axis(['Feature 1', 'Feature 2']).reset_index(name='Correlation'))

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return rmse, r2

# Initialize session state for model and predictions
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Model Section
if st.sidebar.checkbox('Model'):
    st.subheader('Model')

    # Test Size
    test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20) / 100

    # Split data
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model Selection
    model_option = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

    # Train and Evaluate Model
    if st.button('Train Model'):
        if model_option == "Linear Regression":
            st.session_state.model = LinearRegression()
        else:
            st.session_state.model = RandomForestRegressor(random_state=42)
        
        st.session_state.model.fit(X_train, y_train)
        st.session_state.predictions = st.session_state.model.predict(X_test)
        rmse, r2 = evaluate_model(st.session_state.model, X_test, y_test)
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

    # Plotting Actual vs Predicted values
    if st.checkbox("Show Actual vs Predicted Plot") and st.session_state.predictions is not None:
        fig, ax = plt.subplots()
        ax.scatter(y_test, st.session_state.predictions, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title("Actual vs Predicted Quality")
        st.pyplot(fig)

    # Show Feature Importances for Random Forest
    if model_option == "Random Forest" and st.checkbox("Show Feature Importances"):
        if st.session_state.model is not None:
            importances = st.session_state.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            fig, ax = plt.subplots()
            ax.set_title("Feature Importances")
            ax.bar(range(X_train.shape[1]), importances[indices], align="center")
            ax.set_xticks(range(X_train.shape[1]))
            ax.set_xticklabels(X.columns[indices], rotation=90)
            st.pyplot(fig)


            st.subheader("Random Forest Hyperparameter Tuning")

            # Parameters for Grid Search
            n_estimators = st.slider("Number of Estimators", 100, 500, 100)
            max_depth = st.slider("Max Depth", 10, 50, 10)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
            
            if st.button("Tune Hyperparameters"):
                param_grid = {
                    'n_estimators': [n_estimators],
                    'max_depth': [max_depth],
                    'min_samples_split': [min_samples_split]
                }
                
                grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                
                best_params = grid_search.best_params_
                st.write("Best Parameters:", best_params)
                
                # Evaluate the best model
                best_rf_model = grid_search.best_estimator_
                tuned_rmse, tuned_r2 = evaluate_model(best_rf_model, X_test, y_test)
                st.write(f"Tuned RMSE: {tuned_rmse:.2f}")
                st.write(f"Tuned R² Score: {tuned_r2:.2f}")