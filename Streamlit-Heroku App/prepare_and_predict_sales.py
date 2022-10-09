import pandas as pd
import numpy as np
import xgboost as xgb

# load the model
model = xgb.XGBRegressor()
model.load_model("model_xgb_final.bin")

# Prepare dataset and model features to be used 
def prepare_updated_dataset(model_features, df):
    # remove NaNs
    df.fillna(0, inplace=True)
    
    #make data types consistent    
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].astype(int)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].astype(int)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].astype(int)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].astype(int)
    df['PromoInterval'] = df['PromoInterval'].astype(str)
    df['SchoolHoliday'] = df['SchoolHoliday'].astype(float)
    df['StateHoliday'] = df['StateHoliday'].astype(str)    
          
    # Use these properties directly
    model_features.extend(['Store', 'CompetitionDistance', 'Promo', 'SchoolHoliday'])

    # encode these model_features
    model_features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    df.StoreType.replace(mappings, inplace=True)
    df.Assortment.replace(mappings, inplace=True)
    df.StateHoliday.replace(mappings, inplace=True)

    model_features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['WeekOfYear'] = df.Date.dt.weekofyear
    
    
    model_features.append('CompetitionOpen')
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    # Promo open time in months
    model_features.append('PromoOpen')
    df['PromoOpen'] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek) / 4.0
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df.loc[df.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate whether sales on a day are in promo interval
    model_features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['monthStr'] = df.Month.map(month2str)
    df.loc[df.PromoInterval == 0, 'PromoInterval'] = ''
    df['IsPromoMonth'] = 0
    for interval in df.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                df.loc[(df.monthStr == month) & (df.PromoInterval == interval), 'IsPromoMonth'] = 1

    return df,model_features
    
# prediction
def predict_sales(df,model_features):
    sales_p = model.predict(df[model_features])
    return np.expm1(sales_p)