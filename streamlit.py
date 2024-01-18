# streamlit run streamlit.py

import streamlit as st
import pandas as pd
import numpy as np

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from math import sqrt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold

# Neural Network
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

# To resolve data imbalance
from imblearn.over_sampling import SMOTE
from collections import Counter


# ------------------------------- Data Preprocessing -------------------------------
df = pd.read_csv('winequality-red.csv')

# Feature selection and data splitting
bins = (2, 6, 8) # 2-6 bad, 6-8 good
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

# Assign labels: bad = 0, good = 1 
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality']) 

X = df.drop('quality', axis=1)
y = df['quality']

# Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Standard scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ------------------------------- Functions -------------------------------
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return rmse, r2

def plot_histograms(df):
    for column in df.columns[:-1]:  # excluding target variable
        st.subheader(f'{column}')
        fig, ax = plt.subplots(figsize=(10, 5))
        df[column].hist(bins=20, ax=ax)
        st.pyplot(fig)

def plot_box_plots(df):
    for column in df.columns[:-1]:  # excluding target variable
        st.subheader(f'{column}')
        fig, ax = plt.subplots(figsize=(10, 5))
        df.boxplot(column=column, ax=ax)
        st.pyplot(fig)


# ------------------------------------------------- Streamlit -------------------------------------------------
# Sidebar
st.sidebar.header('Menù')

# Title
st.title('Wine Quality Prediction App')

# ------------------------------- Dataframe Section -------------------------------
if st.sidebar.checkbox('DataFrame'):
    st.title('Dataframe')
    st.write(df)
    st.write('Shape: ', df.shape)

    # Columns types
    st.subheader('Columns types')
    st.write(df.dtypes.rename("Type"))

    # Dataframe description
    st.subheader('Dataframe Description')
    st.write(df.describe().T)

    # Target variable
    st.subheader('Target variable distribution')
    fig, ax = plt.subplots(figsize=(10, 5))
    df['quality'].value_counts().sort_index(ascending=True).plot(kind='bar', ax=ax, color=['red', 'green'])
    ax.set_xlabel('Quality')
    ax.set_ylabel('Count')
    plt.xticks(rotation='horizontal')
    st.pyplot(fig)
    st.write(df['quality'].value_counts().sort_index(ascending=True))

    # Add options to display histograms and box plots
    # if st.checkbox('Show Histograms of Features'):
    #     plot_histograms(df)

    if st.checkbox('Show Box Plots of Features'):
        plot_box_plots(df)

# ------------------------------- Correlation Matrix section -------------------------------
if st.sidebar.checkbox('Correlation Matrix'):
    st.subheader('Correlation Matrix')
    sns.heatmap(df.corr(), annot=True, fmt='.1f')
    st.pyplot(plt)

    st.write("The **most correlated** features are: \n")
    st.write(df.corr().unstack().sort_values(ascending=True).drop_duplicates().tail(6)[:-1].rename_axis(['Feature 1', 'Feature 2']).reset_index(name='Correlation'))

# ------------------------------- Model Section -------------------------------

if st.sidebar.checkbox('Model'):
    st.subheader('Model')

    # Test size
    test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20) / 100

    # Split data
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model selection
    model_option = st.selectbox("Select Model", ["Random Forest", "SVM", "Neural Network"])

    # Train and evaluate model
    if st.button('Train Model'):
        if model_option == "Random Forest":
            st.session_state.model = RandomForestClassifier(n_estimators=200)
            st.session_state.model.fit(X_train, y_train)
            pred_rfc = st.session_state.model.predict(X_test)
            importances = st.session_state.model.feature_importances_
            indices = np.argsort(importances)
            features = X_train.columns
            st.bar_chart(pd.DataFrame(importances[indices], index=features[indices]))

            
        elif model_option == "SVM":
            st.session_state.model = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
            st.session_state.model.fit(X_train, y_train)
            pred_svc = st.session_state.model.predict(X_test)


        elif model_option == "Neural Network":
            # Normalize the feature data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Convert target variable to categorical (one-hot encoding)
            y_encoded = to_categorical(y)

            # K-Fold Cross Validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train, test in kfold.split(X_scaled, y_encoded):
                # Create model
                model = Sequential()
                model.add(Dense(64, input_shape=(X_scaled.shape[1],), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                model.add(Dense(32, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                model.add(Dense(y_encoded.shape[1], activation='softmax'))

                # Compile model
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # Train model
                model.fit(X_scaled[train], y_encoded[train], epochs=20, batch_size=64, verbose=0)

                # Evaluate the model
                scores = model.evaluate(X_scaled[test], y_encoded[test], verbose=0)
                cv_scores.append(scores[1] * 100)

            # Average CV score
            avg_cv_score = np.mean(cv_scores)
            st.write(f"Cross-Validated Neural Network Model Accuracy: {avg_cv_score:.2f}%")

        if model_option != "Neural Network":
            st.session_state.predictions = st.session_state.model.predict(X_test)
            rmse, r2 = evaluate_model(st.session_state.model, X_test, y_test)
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"R² Score: {r2:.2f}")
            st.write(f"Accuracy: {accuracy_score(y_test, st.session_state.predictions):.2f}")

    # ------------ commented because the results are the same as without hyperparameter tuning ------------
    # Show Feature Importances for Random Forest 
    # if model_option == "Random Forest" and st.checkbox("Random Forest Hyperparameter Tuning"):
    #     if st.session_state.model is not None:
    #         importances = st.session_state.model.feature_importances_
    #         indices = np.argsort(importances)[::-1]


    #         st.subheader("Random Forest Hyperparameter Tuning")

    #         # Parameters for Grid Search
    #         n_estimators = st.slider("Number of Estimators", 100, 500, 100)
    #         max_depth = st.slider("Max Depth", 10, 50, 10)
    #         min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
            
    #         if st.button("Tune Hyperparameters"):
    #             param_grid = {
    #                 'n_estimators': [n_estimators],
    #                 'max_depth': [max_depth],
    #                 'min_samples_split': [min_samples_split]
    #             }
                
    #             grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    #             grid_search.fit(X_train, y_train)
                
    #             best_params = grid_search.best_params_
    #             st.write("Best Parameters:", best_params)
                
    #             # Evaluate the best model
    #             best_rf_model = grid_search.best_estimator_
    #             tuned_rmse, tuned_r2 = evaluate_model(best_rf_model, X_test, y_test)
    #             st.write(f"Tuned RMSE: {tuned_rmse:.2f}")
    #             st.write(f"Tuned R² Score: {tuned_r2:.2f}")
    #             st.write(f"Tuned Accuracy: {accuracy_score(y_test, best_rf_model.predict(X_test)):.2f}")


# ------------------------------- Try Prediction Section -------------------------------
@st.cache_data
def get_data():
    data = pd.read_csv('winequality-red.csv')
    return data

# Preprocess the data for Data Imbalance
@st.cache_data
def preprocess_data(data):
    bins = (2, 6, 8)  # Define bins as 2-6: bad, and 6-8: good
    group_names = ['bad', 'good']
    data['quality'] = pd.cut(data['quality'], bins=bins, labels=group_names)
    label_quality = LabelEncoder()
    data['quality'] = label_quality.fit_transform(data['quality'])
    return data

# Model Training - SVM
# @st.cache_data
# def train_model(X_train, y_train):
#     model = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
#     model.fit(X_train, y_train)
#     return model

# Model Training - Random Forest
@st.cache_data()
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def get_user_input():
    df = get_data()
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

# Main
def main():
    st.title("Wine Quality Prediction App")
    st.subheader("Data Preprocessing and Model Training")

    # Getting user input
    user_input = get_user_input()  # This should be at the beginning of the main function

    # Display the user input features
    st.subheader('User Input parameters')
    st.write(user_input)

    data = get_data()
    processed_data = preprocess_data(data)

    # Displaying class distribution before SMOTE
    st.write('Original dataset shape:', Counter(processed_data['quality']))

    # Split the data
    X = processed_data.drop('quality', axis=1)
    y = processed_data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    # Displaying class distribution after SMOTE
    st.write('Resampled dataset shape:', Counter(y_train_res))

    # Training the model
    model = train_model(X_train_res, y_train_res)

    # Model prediction
    if st.button('Predict'):
        # Apply the same preprocessing to the user input as to the training data
        scaler = StandardScaler().fit(X_train)
        scaled_user_input = scaler.transform(user_input)
        prediction = model.predict(scaled_user_input)
        prediction_text = 'good' if prediction[0] == 1 else 'bad'
        st.subheader('Prediction')
        st.write(f'The wine is of {prediction_text} quality')

if st.sidebar.checkbox('Try Prediction'):
    main()