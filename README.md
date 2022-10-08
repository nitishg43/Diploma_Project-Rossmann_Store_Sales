Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

Dataset 

Data is taken from Kaggle from the following link:

https://www.kaggle.com/competitions/rossmann-store-sales/data

Dataset properties 

Dataset is approximately of size 40 MB and consists of the following files:
train.csv - historical data including Sales
test.csv - historical data excluding Sales
store.csv - supplemental information about the stores

features in train.csv	- store, day of week, date, sales, customers, open, promo, state holiday, school holiday
features in test.csv	- id, store, dayofweek, date, open, promo, state holiday, school holiday	
features in store.csv	- store, storetype, assortment, competition distance, competition open since month, promo2, promo2since week, promo2since year, promo interval	

While training our model, only those entries where sales are greater than zero and only entries for open stores have been considered.

The input 'Date' feature has been split into day, month, and year and these are passed as inputs to our model. First Date column is converted using to_datetime function of pandas and later, the mentioned other features are extracted from Date.

Also, to extract more meaningful information from store.csv training file and to make the features related to Promotion and Competition much more meaningful in our dataset, new features 'CompetitionOpen' has been added which gives the number of months the store has been facing competition due to a competitor being open near the store, feature 'PromoOpen' has been added which gives the number of months the store has been running its Promo2 promotion strategy and a feature 'IsPromoMonth' has been added to find out if the sale on a given date is in the duration of a Promo interval.

The final XGBoost Model used to deply our application has RMSPE of 1.06%.

We have used Streamlit and Heroku to deploy our model.

This is the URL of the web application of deployed model.

https://rossmann-sales-pred.herokuapp.com/
