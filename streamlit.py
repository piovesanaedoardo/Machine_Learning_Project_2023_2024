# streamlit app for the project
# streamlit run streamlit.py

# import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from math import sqrt

# Dataset
df = pd.read_csv('winequality-red.csv')
# st.write(df.head())

# Feature selection and Data Splitting
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = RandomForestClassifier()

# Model Training
model.fit(X_train, y_train)

# Model Evaluation
predicitons = model.predict(X_test)

accuracy = accuracy_score(y_test, predicitons)
# st.write(f'Model Accuracy (Random Forest): {accuracy}')

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return rmse, r2

# Function to plot histograms
def plot_histograms(df):
    for column in df.columns[:-1]:  # Excluding the target variable 'quality'
        st.subheader(f'{column}')
        fig, ax = plt.subplots()
        df[column].hist(bins=20, ax=ax)
        st.pyplot(fig)

# Function to plot box plots
def plot_box_plots(df):
    for column in df.columns[:-1]:  # Excluding the target variable 'quality'
        st.subheader(f'{column}')
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
    model_option = st.selectbox("Select Model", ["Random Forest", "SVM", "AdaBoost"])

    # Train and Evaluate Model
    if st.button('Train Model'):
        if model_option == "Random Forest":
            st.session_state.model = RandomForestClassifier()
            importances = model.feature_importances_
            indices = np.argsort(importances)
            features = X_train.columns
            st.bar_chart(pd.DataFrame(importances[indices], index=features[indices]))

        if model_option == "SVM":
            st.session_state.model = SVC()
            importances = model.feature_importances_
            indices = np.argsort(importances)
            features = X_train.columns
            st.bar_chart(pd.DataFrame(importances[indices], index=features[indices]))


        if model_option == "AdaBoost":
            st.session_state.model = AdaBoostClassifier()
            importances = model.feature_importances_
            indices = np.argsort(importances)
            features = X_train.columns
            st.bar_chart(pd.DataFrame(importances[indices], index=features[indices]))
        
        st.session_state.model.fit(X_train, y_train)
        st.session_state.predictions = st.session_state.model.predict(X_test)
        
        # Evaluate the model
        rmse, r2 = evaluate_model(st.session_state.model, X_test, y_test)
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

    # Show Feature Importances for Random Forest
    if model_option == "Random Forest" and st.checkbox("Random Forest Hyperparameter Tuning"):
        if st.session_state.model is not None:
            importances = st.session_state.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
    # if model_option == "Random Forest" and st.checkbox("Show Feature Importances"):
    #     if st.session_state.model is not None:
    #         importances = st.session_state.model.feature_importances_
    #         indices = np.argsort(importances)[::-1]
    #         fig, ax = plt.subplots()
    #         ax.set_title("Feature Importances")
    #         ax.bar(range(X_train.shape[1]), importances[indices], align="center")
    #         ax.set_xticks(range(X_train.shape[1]))
    #         ax.set_xticklabels(X.columns[indices], rotation=90)
    #         st.pyplot(fig)


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
                
                grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                
                best_params = grid_search.best_params_
                st.write("Best Parameters:", best_params)
                
                # Evaluate the best model
                best_rf_model = grid_search.best_estimator_
                tuned_rmse, tuned_r2 = evaluate_model(best_rf_model, X_test, y_test)
                st.write(f"Tuned RMSE: {tuned_rmse:.2f}")
                st.write(f"Tuned R² Score: {tuned_r2:.2f}")

# Function to get user input
def get_user_input():
    fixed_acidity = st.slider('Fixed Acidity', min_value=float(df['fixed acidity'].min()), max_value=float(df['fixed acidity'].max()), value=float(df['fixed acidity'].mean()))
    volatile_acidity = st.slider('Volatile Acidity', min_value=float(df['volatile acidity'].min()), max_value=float(df['volatile acidity'].max()), value=float(df['volatile acidity'].mean()))
    citric_acid = st.slider('Citric Acid', min_value=float(df['citric acid'].min()), max_value=float(df['citric acid'].max()), value=float(df['citric acid'].mean()))
    residual_sugar = st.slider('Residual Sugar', min_value=float(df['residual sugar'].min()), max_value=float(df['residual sugar'].max()), value=float(df['residual sugar'].mean()))
    chlorides = st.slider('Chlorides', min_value=float(df['chlorides'].min()), max_value=float(df['chlorides'].max()), value=float(df['chlorides'].mean()))
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', min_value=float(df['free sulfur dioxide'].min()), max_value=float(df['free sulfur dioxide'].max()), value=float(df['free sulfur dioxide'].mean()))
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', min_value=float(df['total sulfur dioxide'].min()), max_value=float(df['total sulfur dioxide'].max()), value=float(df['total sulfur dioxide'].mean()))
    density = st.slider('Density', min_value=float(df['density'].min()), max_value=float(df['density'].max()), value=float(df['density'].mean()))
    pH = st.slider('pH', min_value=float(df['pH'].min()), max_value=float(df['pH'].max()), value=float(df['pH'].mean()))
    sulphates = st.slider('Sulphates', min_value=float(df['sulphates'].min()), max_value=float(df['sulphates'].max()), value=float(df['sulphates'].mean()))
    alcohol = st.slider('Alcohol', min_value=float(df['alcohol'].min()), max_value=float(df['alcohol'].max()), value=float(df['alcohol'].mean()))

    user_data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    return pd.DataFrame(user_data, index=[0])

# Streamlit main function
def main():
    st.subheader("Please enter the following details to predict the quality of wine")

    # Model selection
    model_type = st.selectbox("Choose the model for prediction", ("Random Forest", "SVM", "AdaBoost"))

    # Load user input
    user_input = get_user_input()

    # Model prediction
    if st.button('Predict'):
        if model_type == 'Random Forest':
            model = RandomForestClassifier()
        elif model_type == 'SVM':
            model = SVC()
        elif model_type == 'AdaBoost':
            model = AdaBoostClassifier()

        # Fit the model (Note: Use your preprocessed training data here)
        model.fit(X_train, y_train)

        # Make prediction
        prediction = model.predict(user_input)

        # Display result
        st.subheader("Prediction")
        st.write("Based on the input, the predicted quality of the wine is: ", prediction[0])
        # st.write(prediction)

if st.sidebar.checkbox('Try Prediction'):
    main()