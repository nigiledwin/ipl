import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRFRegressor

# Importing dataframe
try:
    df = pd.read_csv('all_season_details.csv')
except pd.errors.ParserError as e:
    print("ParserError:", e)

# Columns to keep
columns_to_keep = ['season', 'match_id', 'current_innings', 'innings_id', 'match_name', 'home_team', 'away_team', 'over', 'ball', 'runs', 'isBoundary', 'isWide', 'wicket_id', 'isNoball', 'batsman1_name', 'batsman_runs']
df_filtered = df[columns_to_keep].copy()  # Ensure you're working on a copy

# Adding additional columns
df_filtered['total_runs'] = df.groupby(['match_id', 'innings_id'])['runs'].transform('cumsum').astype(int)
df_filtered['Wkts'] = df_filtered.groupby(['match_id', 'innings_id'])['wicket_id'].transform(lambda x: x.notnull().cumsum()).astype(int)
df_filtered['final_scores'] = df_filtered.groupby(['match_id', 'current_innings'])['runs'].transform('sum')
df_filtered['1st_10overruns'] = df_filtered[df_filtered['over']<=10].groupby(['match_id', 'current_innings'])['runs'].transform('sum')
df_filtered['1st_10overruns'] = df_filtered['1st_10overruns'].fillna(method='ffill')
df_filtered['1st_10overwkts'] = df_filtered[df_filtered['over']<=10].groupby(['match_id', 'current_innings'])['wicket_id'].transform('count')
df_filtered['1st_10overwkts'] = df_filtered['1st_10overwkts'].fillna(method='ffill')
df_filtered['bowling_team'] = df_filtered.apply(lambda row: row['away_team'] if row['home_team'] == row['current_innings'] else row['home_team'], axis=1)
df_filtered['Is_bat_home_team'] = df_filtered.apply(lambda row: 'Yes' if row['home_team'] == row['current_innings'] else 'No', axis=1)
print(df_filtered.head())



# Assuming df is your DataFrame containing the match data

# Sort the DataFrame by season and match_id to ensure chronological order
df_sorted = df_filtered.sort_values(by=['season', 'match_id'])

# Group the DataFrame by the team
grouped = df_sorted.groupby('current_innings')

# Calculate the rolling mean of the final scores for the last 10 matches for each team
rolling_mean = grouped['1st_10overruns'].rolling(window=10, min_periods=1).mean()

# Reset the index to align the rolling mean values with the original DataFrame
rolling_mean = rolling_mean.reset_index(level=0, drop=True)

# Assign the rolling mean values to a new column in the original DataFrame
df_filtered['last_10_matches_mean_10overscore'] = rolling_mean


columns_to_select_ppruns=['current_innings','bowling_team','Is_bat_home_team','over', 'ball','total_runs','Wkts','1st_10overruns']
df_pp_ml=df_filtered[columns_to_select_ppruns]
df_pp_ml.rename(columns={'current_innings': 'batting_team'}, inplace=True)
df_final=df_pp_ml[df_pp_ml['over']<=10]
print(df_final.tail(5))

df_final.loc[df_final['batting_team'] == 'KXIP', 'batting_team'] = 'PBKS'
df_final.loc[df_final['bowling_team'] == 'KXIP', 'bowling_team'] = 'PBKS'
df_final.loc[df_final['batting_team'] == 'GL', 'batting_team'] = 'GT'
df_final.loc[df_final['bowling_team'] == 'GL', 'bowling_team'] = 'GT'
df_final.loc[df_final['batting_team'] == 'PWI', 'batting_team'] = 'RPS'
df_final.loc[df_final['bowling_team'] == 'PWI', 'bowling_team'] = 'RPS'
df_final = df_pp_ml[(df_pp_ml['1st_10overruns'] >= 15) & (df_pp_ml['1st_10overruns'] <= 150)]


X=df_final.drop(columns=['1st_10overruns'])
y=df_final['1st_10overruns']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

# Define the ColumnTransformer for one-hot encoding
trf1 = ColumnTransformer([
    ('team_ohe', OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first'), [0, 1, 2])
], remainder='passthrough')

# Define XGBoost parameters
xgb_params = {
    'subsample': 1.0,
    'reg_lambda': 0.1,
    'reg_alpha': 0.5,
    'n_estimators': 300,
    'min_child_weight': 1,
    'max_depth': 7,
    'learning_rate': 0.5,
    'gamma': 0,
    'colsample_bytree': 0.6
}

#defining all base models

lr=LinearRegression()
rf=RandomForestRegressor()
ridge=Ridge(alpha=0.1)
svr=SVR(C=1.0, epsilon=0.2)
gb=GradientBoostingRegressor()
xgb_model =xgb.XGBRFRegressor(**xgb_params)


# Define other models for the voting ensemble, stacking and Blending
voting_base_models = [
    ('rf', rf),
    ('lr', lr),
    ('ridge', ridge),
    ('svr', svr),
    ('gb', gb),
]

stacking_base_models = [
    ('rf', rf),
    ('lr', lr),
    ('ridge', ridge),
    ('svr', svr),
    ('gb', gb),
    ('xgb', xgb_model)
]

blending_base_models = [
    ('rf', rf),
    ('lr', lr),  # Ridge regression with regularization (alpha = 0.1)
    ('ridge', ridge),  # Another Ridge regression example
    ('svr', svr),  # You can also adjust SVR parameters
    ('gb', gb),
    ('xgb', xgb_model)
]

# Add XGBoost model to the list of models
voting_base_models.append(('xgb', xgb_model))

# Create the voting regressor
voting_regressor = VotingRegressor(estimators=voting_base_models)



# Define meta-model for stacking
meta_model_stacking = LinearRegression()  # You can choose any appropriate meta-model
#Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=stacking_base_models,
    final_estimator=meta_model_stacking,
    cv=5  # Number of folds for cross-validation
)

# Create the pipeline including the stacking regressor


# Define meta-model for blending
meta_model_blending = LinearRegression()  # You can choose any appropriate meta-model

# Create the stacking regressor
blending_regressor = StackingRegressor(
    estimators=blending_base_models,
    final_estimator=meta_model_blending
)

# Create the pipelines for each models
pipe_lr=Pipeline([
    ('trf1', trf1),
    ('trlr', lr)
])
pipe_rf=Pipeline([
    ('trf1', trf1),
    ('trlr', rf)
])
pipe_svr=Pipeline([
    ('trf1', trf1),
    ('trlr', svr)
])
pipe_ridge=Pipeline([
    ('trf1', trf1),
    ('trlr', ridge)
])
pipe_xgb=Pipeline([
    ('trf1', trf1),
    ('trlr', xgb_model)
])
pipe_gb=Pipeline([
    ('trf1', trf1),
    ('trlr', gb)
])

pipe_voting_regressor = Pipeline([
    ('trf1', trf1),
    ('voting', voting_regressor)
])

pipe_stacking_regressor= Pipeline([
    ('trf1', trf1),
    ('stacking', stacking_regressor)
])
pipe_blending = Pipeline([
    ('trf1', trf1),
    ('blending', blending_regressor)
])


models=[pipe_lr,pipe_rf,pipe_ridge,pipe_svr,pipe_xgb,pipe_xgb,pipe_voting_regressor,pipe_stacking_regressor,pipe_blending]

# Now you can fit the pipeline to your data and use it for prediction
from sklearn.metrics import r2_score
new_data = pd.DataFrame({
    'batting_team': ['SRH'],      # Example: Batting team
    'bowling_team': ['MI'],       # Example: Bowling team
    'Is_bat_home_team': ['Yes'],         # Example: 1st powerplay runs
    'over': [6.0],          # Example: 1st powerplay wickets
    'ball':[6],
    'total_runs':[70],
    'Wkts':[1]


})

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Model', 'Training R^2', 'Training RMSE', 'Test R^2', 'Test RMSE','New_data_pred'])
print(results_df)
def pridict(models):
    for model_name, model in models:

        # Train the pipeline
        model.fit(X_train, y_train)

        # Predict on the training data
        y_pred_train = model.predict(X_train)

        # Calculate R^2 score and RMSE on the training data
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(((y_pred_train - y_train) ** 2).mean())

        # Predict on the test data
        y_pred_test = model.predict(X_test)

        # Calculate R^2 score and RMSE on the test data
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(((y_pred_test - y_test) ** 2).mean())
        y_pred = np.round(model.predict(new_data))


        # Append the results to the DataFrame
        results_df.loc[len(results_df)] = [model_name, r2_train, rmse_train, r2_test, rmse_test,y_pred]
# Call the function to evaluate the models and store the results
pridict([('Linear Regression', pipe_lr),
         ('Random Forest', pipe_rf),
         ('Ridge Regression', pipe_ridge),
         ('SVR', pipe_svr),
         ('XGBoost', pipe_xgb),
         ('Gradient Boosting', pipe_gb),
         ('Voting Regressor', pipe_voting_regressor),
         ('Stacking Regressor', pipe_stacking_regressor),
         ('Blending Regressor', pipe_blending)])



import pickle
import joblib
# Define a dictionary to store the pipeline models
pipeline_models = {
    'Linear Regression': pipe_lr,
    'Random Forest': pipe_rf,
    'Ridge Regression': pipe_ridge,
    'SVR': pipe_svr,
    'XGBoost': pipe_xgb,
    'Gradient Boosting': pipe_gb,
    'Voting Regressor': pipe_voting_regressor,
    'Stacking Regressor': pipe_stacking_regressor,
    'Blending Regressor': pipe_blending
}

# Export each pipeline model
for model_name, pipeline_model in pipeline_models.items():
    filename = f'{model_name}.pkl'  # Change the extension to .pkl
    joblib.dump(pipeline_model, filename)
    print(f'Saved {model_name} model as {filename}')

# Create a DataFrame with X_test, y_test, and y_pred
result_df = pd.DataFrame({
    'X_test': X_test.values.tolist(),
    'y_test': y_test.tolist(),
    'y_pred': pipe_blending.predict(X_test).tolist()  # Assuming 'New_data_pred' column contains the predictions
})

print(result_df)    