import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import randint,uniform

df1 = pd.read_csv('Plant_2_Generation_Data.csv',encoding='latin1')
df2 = pd.read_csv('Plant_2_Weather_Sensor_Data.csv',encoding='latin1')

df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'])
df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'])

df = pd.merge(df1,df2,on='DATE_TIME',how='inner')

# Exploratry Data Analysis (EDA)

null_values = df.isnull().sum()

daily_power_trend = df.groupby('DATE_TIME')['DC_POWER'].sum().reset_index()

day_to_plot = daily_power_trend['DATE_TIME'].dt.date.iloc[0]
daily_plot_data = daily_power_trend[daily_power_trend['DATE_TIME'].dt.date == day_to_plot]

numerical_df = df[['DC_POWER', 'AC_POWER', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
correlations = numerical_df.corr()

plt.figure()
sns.lineplot(x='DATE_TIME',y='DC_POWER',data=daily_plot_data,color='skyblue')

plt.title(f'Total DC power generation {day_to_plot}',fontsize=16)
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Power Generated (DC)',fontsize=12)

plt.figure()
sns.heatmap(correlations,annot=True,cmap='viridis')


print('Null values :-', null_values)


#Feature Engineering

df.set_index('DATE_TIME',inplace=True)
df.sort_index(inplace=True)

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

df['IRRADIATION_LAG_1'] = df['IRRADIATION'].shift(1)

df.dropna(inplace=True)

categorical_features = ['hour','day_of_week','month']
numerical_features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'IRRADIATION_LAG_1']

preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features )

    ]
)

x = df[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'IRRADIATION_LAG_1','hour', 'day_of_week', 'month']]
y = df['DAILY_YIELD']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Model Training and Evaluation


#Logistic Regression

lg_pipeline = Pipeline(
    steps = [
        ('preprocessor',preprocessor),
        ('regessor',LinearRegression())
    ]
)

lg_pipeline.fit(x_train,y_train)

y_pred_lg = lg_pipeline.predict(x_test)

r2_score_lg = r2_score(y_test,y_pred_lg)
mae_lg = mean_absolute_error(y_test,y_pred_lg)
mse_lg = mean_squared_error(y_test,y_pred_lg)


print('--- Linear regression Regression Evaluation ---')
print(f' \n R2 SCORE of Linear Regression :- {r2_score_lg}')
print(f'Mean Absolute error of Linear regression :- {mae_lg}')
print(f'Mean squared error of Linear regression :- {mse_lg}')

plot_data_lg = pd.DataFrame({
    'Actual' : y_test,
    'Predicted' : y_pred_lg
}, index=y_test.index)

plot_data_lg.sort_index(inplace=True)

plt.figure(figsize=(10,8))
sns.lineplot(data=plot_data_lg,x=plot_data_lg.index,y='Actual',label='Actual Daily Yield',color='red')

sns.lineplot(data=plot_data_lg,y='Predicted',x=plot_data_lg.index,label='Linear Regession Predicted Daily Yields',color='yellow')

plt.title('Predicted vs Actual Daily Solar Yield',fontsize=18)
plt.xlabel('Date and Time',fontsize=14)
plt.ylabel('Daily Yield (Wh)',fontsize=14)

#Random Forest

rf_pipeline = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor',RandomForestRegressor(random_state=42))
    ]
)

rf_pipeline.fit(x_train,y_train)

y_pred_rf = rf_pipeline.predict(x_test)

mse_rf = mean_squared_error(y_test,y_pred_rf)
mae_rf = mean_absolute_error(y_test,y_pred_rf)
r2_score_rf = r2_score(y_test,y_pred_rf)


print('--- Random Forest Regressor Evaluation ---\n')
print(f'Mean Squared Error of Random forest :- {mse_rf}')
print(f'Mean Absolute Error of Random forest :- {mae_rf} ')
print(f'R2 Score of Random forest :- {r2_score_rf}')

#XGBOOST

pipeline_xg = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor',XGBRegressor(use_label_encoder=False,random_state=42))
    ]
)


param_distribution_xg = {
    'regressor__n_estimator' : randint(low=100,high=500),
    'regressor__learning_rate' : uniform(0.01,0.2),
    'regressor__max_depth' : randint(3,10),
    'regressor__subsample' : uniform(0.6,0.4)

}

random_search_xg = RandomizedSearchCV(
    estimator=pipeline_xg,
    param_distributions=param_distribution_xg,
    n_iter=50,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)


random_search_xg.fit(x_train,y_train)

y_pred_xg = random_search_xg.predict(x_test)

mae_xg = mean_absolute_error(y_test,y_pred_xg)
mse_xg = mean_squared_error(y_test,y_pred_xg)
r2_score_xg = r2_score(y_test,y_pred_xg)


print('--- XGB Model Evaluation --- \n')
print(f'Mean squared error of XGBoost {mse_xg}')
print(f'Mean Absolutely error of XGBoost :- {mae_xg}')
print(f' R2 Score of XGboost :- {r2_score_xg} ')

plot_data_xg = pd.DataFrame({
    'Actual' : y_test,
    'Predicted' : y_pred_xg,
},index=y_test.index)

plt.figure(figsize=(11,8))

sns.lineplot(data=plot_data_xg,x=plot_data_xg.index,y='Actual',label='Actual Daily Yield', color='yellow')
sns.lineplot(data=plot_data_xg,x=plot_data_xg.index,y='Predicted',label='XGBoost Prediction of Daily Yield', color='blue')

plt.title('Daily Yield vs XGBoost Predictions',fontsize=14)
plt.xlabel('Date and time',fontsize=14)
plt.ylabel('Daily Yeild',fontsize=14)



#Feature Importance Plotting

best_model = random_search_xg.best_estimator_
feature_importance = best_model.named_steps['regressor'].feature_importances_

one_hot_features = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = numerical_features + list(one_hot_features)

feature_importance_df = pd.DataFrame({
    'Feature' : all_features,
    'Importance' : feature_importance
}).sort_values(by='Importance',ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(data=feature_importance_df.head(10),x='Importance',y='Feature',palette='rainbow')
plt.title('Top 15 most important features')
plt.xlabel('Importance Score')
plt.ylabel('Feature')


#Daily and seasonal patterns
df['DAILY_YIELD'] = df.groupby(df.index.date)['DAILY_YIELD'].transform('sum')

#Daily Patterns of hours
df['hour'] = df.index.hour
daily_pattern = df.groupby('hour')[['DAILY_YIELD','IRRADIATION']].mean()

plt.figure(figsize=(10,8))
sns.lineplot(data=daily_pattern,x=daily_pattern.index,y='IRRADIATION',label='Average Irridiation',color='orange')
plt.twinx()
sns.lineplot(data=daily_pattern,x=daily_pattern.index,y='DAILY_YIELD',label='Average Daily yield',color='blue')
plt.title('Average Daily Yield and Irradiation by Hour', fontsize=16)
plt.xlabel('Hour of day')
plt.ylabel('Average Daily Yield (Wh)')

df['month'] = df.index.month
seasonal_pattern = df.groupby('month')[['DAILY_YIELD', 'IRRADIATION']].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(data=seasonal_pattern, x=seasonal_pattern.index, y='IRRADIATION', label='Average Irradiation', color='orange')
plt.twinx()
sns.lineplot(data=seasonal_pattern, x=seasonal_pattern.index, y='DAILY_YIELD', label='Average Daily Yield', color='green', linestyle='--')
plt.title('Average Daily Yield and Irradiation by Month', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Daily Yield (Wh)', fontsize=12)


# Final plot
plt.figure(figsize=(11, 8))
sns.lineplot(data=plot_data_xg, x=plot_data_xg.index, y='Actual', label='Actual Daily Yield', color='blue')
sns.lineplot(data=plot_data_xg, x=plot_data_xg.index, y='Predicted', label='XGBoost Prediction of Daily Yield', color='red', linestyle='--')
plt.title('Daily Yield vs XGBoost Predictions', fontsize=14)
plt.xlabel('Date and time', fontsize=14)
plt.ylabel('Daily Yield (Wh)', fontsize=14)
plt.legend()
plt.tight_layout()



plt.show()


#Saving Model and features for Frontend, Backend
import joblib
joblib.dump(best_model,'Best_model_power_prediction.joblib')
joblib.dump(all_features,'Features.joblib')


