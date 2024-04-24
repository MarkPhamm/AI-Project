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
import data_cleaning
import eda



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
    st.write("This is the content of Tab 2")

# Create tabs
tabs = ["EDA", "ANN Model"]
selected_tab = st.selectbox("Select Tab", tabs)

# Display selected tab content
if selected_tab == "EDA":
    tab1()
elif selected_tab == "ANN Model":
    tab2()
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


