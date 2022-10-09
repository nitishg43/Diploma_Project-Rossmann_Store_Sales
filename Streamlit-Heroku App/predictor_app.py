import streamlit as st
import pandas as pd
from prepare_and_predict_sales import prepare_updated_dataset, predict_sales
import datetime
import numpy as np
import time

# set title
st.title('Rossmann Store Daily Sales Predictor')

# set subheader
st.header('Forecast daily Sales for next six weeks : ')

# input values from user
# date
#date = st.date_input(label = 'Enter Date of Interest', value = datetime.date(2015, 8, 1), min_value = datetime.date(2015, 8, 1))
today = datetime.date.today()
date = st.date_input(label = 'Enter Date of Interest', value = today, min_value = datetime.date(2013, 1, 1))

# store id
storeid = st.number_input('Enter Store Id', min_value=1, max_value=1115, value=1)

# promo
promo_inp = st.checkbox('Will Store be running a Promo?', value = False)
if promo_inp:
    promo = 1
else:
    promo = 0

# state holiday feature
state_inp = st.selectbox('Will there be a State Holiday?',('Public holiday', 'Easter holiday', 'Christmas', 'None'), index = 3)
mapping = {'Public holiday': 'a', 'Easter holiday':'b', 'Christmas':'c', 'None':'0'}
state_holiday = mapping[state_inp]

# school holiday feature
school_inp = st.selectbox('Will there be a School Holiday?',('Yes', 'No'), index = 1)
mapping1 = {'Yes': '1', 'No':'0'}
school_holiday = mapping1[school_inp]

pred_button = st.button('Predict')
sales_prediction = None
if pred_button:

        
    # creating df
    user_df = pd.DataFrame({'Date':date, 'Store':storeid, 'Promo': promo,'StateHoliday':state_holiday, 'SchoolHoliday' : school_holiday}, index = [0])

    # merging with store constants
    #store = pd.read_csv('/content/drive/MyDrive/DiplomaProject/store.csv')
    store = pd.read_csv('store.csv')

    final_df = user_df.merge(store, on = 'Store')
    model_features = []
    final_df,model_features = prepare_updated_dataset(model_features,final_df)
    
    sales_prediction = predict_sales(final_df,model_features)
    sales_df= pd.DataFrame({'Sales(€) ': np.round(sales_prediction, 4)})
    st.write('The input features are: \n')
    #st.write(final_df)
    # To hide index on page display,CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    final_df=final_df[['Date','Store','Promo','StateHoliday','SchoolHoliday','StoreType','Assortment','CompetitionDistance']]
    st.table(final_df)    
    st.write(f'\nThe predicted Sales for Store', storeid,' on ',date, ' is')
    st.table(sales_df)
    #st.write(f"\nThe predicted Sales for Store {storeid} on {date} is {np.round(sales_prediction, 4)} €")

#Check if the User wants to see the Business problem that this project is trying to solve
project_overview = st.checkbox( 
                        label = 'Show me the Business Problem that this Project addresses.',
                        value = False
                            )

# Display the Business problem that this project is trying to solve
if project_overview:
     
    
    # progress bar for display aesthetics
    progress_bar = st.progress(0)
    
    for percent_complete in range(100):
        time.sleep(0.0000001)
        progress_bar.progress( percent_complete + 1 )

    st.markdown( """# **Rossmann Store Sales Prediction**
## Introduction :
Predicting sales performance is one of the main challenges faced by every business. Sales forecasting which refers to the process of estimating demand of products over specific set of time in future is important, as the demand for products keeps changing from time to time and it is crucial for business firms to predict customer demands to offer the right products at the right time. Sales forecasting also helps to maintain adequate inventory of products and thus, improving their financial performance. 
## Problem Statement : 
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.
## Dataset used in building the model:
Data is taken from Kaggle from the following link:
https://www.kaggle.com/competitions/rossmann-store-sales/data
## Business solution that this project delivers : 
For each store, the daily sales predictions for the next six weeks."""
)