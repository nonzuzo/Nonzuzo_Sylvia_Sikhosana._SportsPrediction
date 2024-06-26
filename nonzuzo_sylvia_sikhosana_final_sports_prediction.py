# -*- coding: utf-8 -*-
"""Nonzuzo_Sylvia_Sikhosana_Final_Sports_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GsAEDsXng6eUZO_GWS-P2rAK75-lgpRu

Question 1
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

#loading the data sets
#from google.colab import drive
#drive.mount('/content/drive')
#test_df= pd.read_csv('/content/drive/My Drive/players_22.csv',low_memory=False)# testing (players_22)
train_df= pd.read_csv('male_players.csv',low_memory= False)# training(males)}

train_df.head()

train_df.columns.tolist()

print(train_df.info())

print(train_df.isnull().sum(), "\n\n")

# Handling the missing values , drop all that have 30% of null values
threshold_train = len(train_df) * 0.3

train_df = train_df.dropna(thresh=len(train_df) - threshold_train, axis=1)

train_df.info()

train_df

print(train_df['age'])

# Columns to drop
columns_to_drop = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam',
                   'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm',
                   'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb']

columns_to_drop_train = ['player_url', 'fifa_update_date', 'short_name', 'long_name', 'club_joined_date','real_face','nationality_name', 'body_type','club_name','preferred_foot', 'body_type', 'gk','league_name',]

columns_to_drop_sum = columns_to_drop + columns_to_drop_train

columns_to_drop_sum.extend(['player_id',	'fifa_version',	'fifa_update', 'dob', 'player_face_url','league_id','potential', 'club_team_id', 'club_position',
 'club_jersey_number',
 'club_contract_valid_until_year',])


train_df.drop(columns=columns_to_drop_sum, inplace=True, errors='ignore')

train_df['work_rate']

train_df.head()

# separating the numeric and catergorical for train_df
# Separate the numeric and Non- numeric features
numeric_data_train_df= train_df.select_dtypes(include = np.number)

non_numeric_train_df = train_df.select_dtypes(include =['object'])

numeric_data_train_df.info()

non_numeric_train_df.info()

numeric_data_train_df.info()

numeric_data_train_df.columns

# Fill Missing numerical Values for both the train_df and test_df
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

numeric_data_train_df= pd.DataFrame(imputer.fit_transform(numeric_data_train_df),columns=numeric_data_train_df.columns)

numeric_data_train_df['mentality_composure']

from sklearn.preprocessing import LabelEncoder

# Create a function to apply label encoding
def label_encode(df, columns):
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

# List of categorical columns
categorical_columns_train = non_numeric_train_df.columns

# Encode categorical columns in the training dataset
categorical_data_train_df, train_label_encoders = label_encode(non_numeric_train_df, categorical_columns_train)

# Encode categorical columns in the test dataset

# Verifying the encodings are different (for illustration purposes)
print(categorical_data_train_df.head())

# putting the numeric and catergorical data together
# Combine training data
merged_train = pd.concat([numeric_data_train_df,categorical_data_train_df], axis=1)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_train= scaler.fit_transform(merged_train)

scaled_train_df = pd.DataFrame(scaled_train, columns=merged_train.columns)

scaled_train_df.head()

"""Question 2"""

dependent_variable=scaled_train_df['overall']

independent_variables = scaled_train_df.drop(columns=['overall']).copy()

correlation_matrix = independent_variables.corrwith(dependent_variable)

# Extract correlations with 'overall' and sort by absolute values
corr_with_target = correlation_matrix.abs().sort_values(ascending=False)

corr_with_target

n_top_features = 10  # Number of top features to select
top_features = corr_with_target[:n_top_features]



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(x=top_features, y = top_features.index, palette='viridis')
plt.xlabel('Features')
plt.ylabel('Correlation with Overall Rating')
plt.title('Top Features with Correlation with Overall Rating')

"""Question 2"""

feature_subsets = scaled_train_df[top_features.index]

feature_subsets.head()

"""Question 3"""

#Question 3
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = feature_subsets
y =train_df['overall']

concat_df = pd.concat([X, y], axis=1)

#randomization of the data
concat_df = concat_df.sample(frac=1, random_state=42).reset_index(drop=True)

concat_df = pd.concat([X, y], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

#Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) #100
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
rf_rmse = np.sqrt(-rf_scores.mean())

rf_model.fit(X_train, y_train)
score = rf_model.score(X_test, y_test)
mse = mean_squared_error(y_test, rf_model.predict(X_test))

print("RMSE: %.4f" % mse)
print("size of prediction: ", len(rf_model.predict(X_test)))
print("prediction: \n", rf_model.predict(X_test))
print("test score: {0:.4f}\n".format(score))

#XGBoost Regressor
import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
xgb_rmse = np.sqrt(-xgb_scores.mean())
xgb_model.fit(X_train, y_train)
score = xgb_model.score(X_test, y_test)
mse = mean_squared_error(y_test, xgb_model.predict(X_test))

print("RMSE: %.4f" % np.sqrt(mse))
print("size of prediction: ", len(xgb_model.predict(X_test)))
print("prediction: \n", xgb_model.predict(X_test))
print("test score: {0:.4f}\n".format(score))

#Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=500, random_state=42, max_depth=4, min_samples_split=2, learning_rate=0.2)
gb_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
gb_rmse = np.sqrt(-gb_scores.mean())
gb_model.fit(X_train, y_train)
score = gb_model.score(X_test, y_test)
mse = mean_squared_error(y_test, gb_model.predict(X_test))

print("RMSE: %.4f" % mse)
print("size of prediction: ", len(gb_model.predict(X_test)))
print("prediction: \n", gb_model.predict(X_test))
print("test score: {0:.4f}\n".format(score))

#DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
param_grid = {
    'max_depth': [2, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 'log2']
}
grid_search = GridSearchCV(dt_model, param_grid, cv=kf, scoring='neg_mean_squared_error')
gs_scores = cross_val_score(dt_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
gs_rmse = np.sqrt(-gs_scores.mean())
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_dt_model = grid_search.best_estimator_

score = best_dt_model.score(X_test, y_test)
mse = mean_squared_error(y_test, best_dt_model.predict(X_test))

print("MSE: %.4f" % mse)
print("size of prediction: ", len(best_dt_model.predict(X_test)))
print("prediction: \n", best_dt_model.predict(X_test))
print("test score: {0:.4f}\n".format(score))

"""Optimization"""

import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error


params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'squared_error'}
improved_gbr = GradientBoostingRegressor(**params)

improved_gbr.fit(X_train,y_train)

score = improved_gbr.score(X_test, y_test)

# calculate the Mean Squared Error
mse = mean_squared_error(y_test, improved_gbr.predict(X_test))

print("MSE: %.4f" % mse)
print("size of prediction: ", len(improved_gbr.predict(X_test)))
print("prediction: \n", improved_gbr.predict(X_test))
print("test score: {0:.4f}\n".format(score))

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.2, 'subsample':0.8}

improved_xgb = xgb.XGBRegressor(**params)

improved_xgb.fit(X_train, y_train)

score = improved_xgb.score(X_test, y_test)

# calculate the Mean Squared Error
mse = mean_squared_error(y_test, improved_xgb.predict(X_test))

print("MSE: %.4f" % mse)
print("size of prediction: ", len(improved_xgb.predict(X_test)))
print("prediction: \n", improved_xgb.predict(X_test))
print("test score: {0:.4f}\n".format(score))

"""Ensembling/ Question 4"""

ensemble = VotingRegressor(estimators=[
    ('improved_xgb', improved_xgb),
    ('rf_model', rf_model),
    ('best_dt_model',best_dt_model),
    ( 'improved_gbr',improved_gbr)
])

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
score = ensemble.score(X_test, y_test)


# Calculate RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)

print("MSE: %.4f" % mse)
print("size of prediction: ", len(ensemble.predict(X_test)))
print("prediction: \n", ensemble.predict(X_test))
print("test score: {0:.4f}\n".format(score))

"""Question 5"""

#testing the with the players_22
#create a function for data procesing ...not traiun ...you are just using it!!!!!!!!!!!!!!!!
test_df= pd.read_csv('players_22.csv',low_memory=False)# testing (players_22)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def label_encode(df, columns):
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    return df, label_encoders

def prepare_data(df):
    # Select relevant features and the dependent variable
    selected_features = ['movement_reactions', 'passing', 'wage_eur', 'mentality_composure',
                         'value_eur', 'dribbling', 'attacking_short_passing', 'mentality_vision',
                         'international_reputation', 'skill_long_passing', 'overall']  # Include 'overall' as dependent variable

    data = df[selected_features].copy()

    numeric_data_test_df = data.select_dtypes(include=np.number)
    non_numeric_test_df = data.select_dtypes(include=['object'])

    # Fill missing values for selected columns
    imputer = SimpleImputer(strategy='mean')
    numeric_data_test_df = pd.DataFrame(imputer.fit_transform(numeric_data_test_df), columns=numeric_data_test_df.columns)

    # List of categorical columns
    categorical_columns_test = non_numeric_test_df.columns

    # Encode categorical columns in the dataset
    categorical_data_test_df, test_label_encoders = label_encode(non_numeric_test_df, categorical_columns_test)

    # Merge numeric and encoded categorical data
    merged_test = pd.concat([numeric_data_test_df, categorical_data_test_df], axis=1)

    scaler = StandardScaler()
    scaled_test = scaler.fit_transform(merged_test)

    # Create a DataFrame with scaled features
    scaled_df = pd.DataFrame(scaled_test, columns=merged_test.columns)

    # Reset index for consistency
    scaled_df.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Combine scaled features with the dependent variable
    final_data = scaled_df.copy()

    # Split into features and target
    X = final_data.drop(columns=['overall'])
    y = merged_test['overall']

    return X, y

# Usage:
# Assuming test_df is your input DataFrame
X_test, y_test = prepare_data(test_df)

print(X_test, y_test)

y_test

# improved_gbr is your trained GradientBoostingRegressor model
score = improved_gbr.score(X_test, y_test)
y_pred_22 = improved_gbr.predict(X_test)

# Print the size of the prediction
print("Size of prediction: ", len(y_pred_22))

# Print the predictions
print("Predictions: \n", y_pred_22)

# Print the test score
print("Test score: {0:.4f}\n".format(score))

score = rf_model.score(X_test,y_test)
y_pred_22 = rf_model.predict(X_test)
print("size of prediction: ", len(rf_model.predict(X_test)))
print("prediction: \n", rf_model.predict(X_test))
print("test score: {0:.4f}\n".format(score))

score = gb_model.score(X_test,y_test)
# Make predictions using the ensemble model
y_pred_22 = gb_model.predict(X_test)
print("size of prediction: ", len(gb_model.predict(X_test)))
print("prediction: \n", gb_model.predict(X_test))
print("test score: {0:.4f}\n".format(score))

score = ensemble.score(X_test, y_test)
# Make predictions using the ensemble model
y_pred_22 = ensemble.predict(X_test)
print("size of prediction: ", len(ensemble.predict(X_test)))
print("prediction: \n", ensemble.predict(X_test))
print("test score: {0:.4f}\n".format(score))

import joblib
joblib.dump(ensemble,'ensemble_model.pkl')

joblib.dump(scaler,'scaler_model.pkl')



