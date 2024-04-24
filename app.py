import streamlit as st
import pandas as pd
import os 
import warnings
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
from openai import OpenAI
# import open_ai
import columns_decode
from sklearn.model_selection import train_test_split
import data_cleaning
import eda
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import lime
import lime.lime_tabular



df = pd.read_csv("NFWBS_PUF_2016_data.csv")
df = data_cleaning.rename_columns(df)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title('Filter')

# Create a selectbox for choosing the score type
score_type = st.sidebar.selectbox(
    'Select Score Type',
    ('Financial well-being scale score (fwb)', 'Financial skill scale score (fs)')
)

# Filter the data based on the selected score type
if score_type == 'Financial well-being scale score (fwb)':
    include_string= 'fwb'
elif score_type == 'Financial skill scale score (fs)':
    include_string = 'fs'

df = data_cleaning.filter_columns(df, include_strings= include_string)
df = data_cleaning.drop_negative_values(df)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def tab1():
    st.write("This is the content of Tab 1")
    
    cat_attribs = [col for col in df.columns if len(df[col].unique()) >= 5 and len(df[col].unique()) <= 7]
    bin_attribs = [col for col in df.columns if len(df[col].unique()) == 2 or len(df[col].unique()) == 1]
    num_attribs = [col for col in df.columns if col not in cat_attribs and col not in bin_attribs and col != 'id']
    
    fig1 = eda.univariate_eda_numerical(df, num_attribs)
    fig2 = eda.univariate_eda_categorical(df, cat_attribs+bin_attribs)
    # fig3 = eda.bivariate_pairplot(df, columns= num_attribs+cat_attribs+bin_attribs)
    fig3 = eda.bivariate_eda_distribution(df, num_attribs, cat_attribs)
    fig4 = eda.multivariate_eda_heatmap(df, bin_attribs+num_attribs+cat_attribs)

    st.header('EDA')
    st.write('Univariate')

    st.plotly_chart(fig1, use_container_width=True, height=400, width=50)
    st.plotly_chart(fig2, use_container_width=True, height=300, width=100)

    st.write('Bivariate')
    # st.pyplot(fig3)
    st.pyplot(fig3)
    st.pyplot(fig4)

def tab2():
    st.write("This is the ANN Model")
    target_column = [col for col in df.columns if 'score' in col.lower()]
    st.write(f"The target we are trying to predict is {target_column[0]}")
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    train = train_set.drop(columns=target_column, errors='ignore')
    train_labels = train_set[target_column]

    test = test_set.drop(columns=target_column, errors='ignore')
    test_labels = test_set[target_column]

        # Define a function to create the model
    def create_model():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(train.shape[1],)),  # Input layer with 64 neurons and ReLU activation
            Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
            Dense(1)  # Output layer with 1 neuron (no activation function for regression)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])  # Using MSE loss for regression and MSE as a metric
        return model

    # Assuming train and train_labels are defined (e.g., train is a pandas DataFrame, train_labels is a pandas Series)
    # Perform k-fold cross-validation
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    mse_scores = []
    for train_indices, val_indices in kfold.split(train):
        X_train, X_val = train.iloc[train_indices], train.iloc[val_indices]
        y_train, y_val = train_labels.iloc[train_indices], train_labels.iloc[val_indices]
        model = create_model()
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        # Evaluate the model on validation data
        mse = model.evaluate(X_val, y_val, verbose=0)[0]
        mse_scores.append(mse)

        # Display MSE scores
    st.write("MSE scores:", mse_scores)
    st.write("Mean MSE:", np.mean(mse_scores))
    st.write("Root Mean Squared Error (RMSE):", np.sqrt(np.mean(mse_scores)))

    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train.values, 
                                                    mode='regression',
                                                    training_labels=train_labels,
                                                    feature_names=train.columns)

    # Generate explanation for a specific data row
    exp = explainer.explain_instance(test.values[5], 
                                    model.predict, 
                                    num_features=len(train.columns))

    # Display explanation in Streamlit
    st.write("Explanation for Test Data Point:")
    st.write("Predicted Value:", exp.predicted_value)
    st.write("True Value:", test_labels.iloc[5])  # Assuming 'test_labels' contains true labels
    st.write("Explanation:")
    # Extract top two features from explanation
    top_features = exp.as_list(2)

    # Display top two features
    st.write("Top Two Features:")
    for feature, value in top_features:
        st.write(f"- {feature}: {value}")

# Create tabs
tabs = ["EDA", "ANN Model"]
selected_tab = st.selectbox("Select Tab", tabs)

# Display selected tab content
if selected_tab == "EDA":
    tab1()
elif selected_tab == "ANN Model":
    tab2()
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


