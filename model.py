​import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os.path
from pandas_ods_reader import read_ods

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

%matplotlib inline

fighter_sheet = "raw_fighter_details"
fight_sheet = "raw_total_fight_data"

fighters_ods = "UFC-DATA-DELETE-LATER/raw_fighter_details.ods"
fights_ods = "UFC-DATA-DELETE-LATER/raw_total_fight_data.ods"

fights_data = read_ods(fights_ods, fight_sheet)
fighters_data = read_ods(fighters_ods, fighter_sheet)

fighters_data.head(50)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

fights_data['R_SIG_STR_pct'] = fights_data['R_SIG_STR_pct'].str.strip('%').astype(float)
fights_data['B_SIG_STR_pct'] = fights_data['B_SIG_STR_pct'].str.strip('%').astype(float)
fights_data['R_TD_pct'] = fights_data['R_TD_pct'].str.strip('%').astype(float)
fights_data['B_TD_pct'] = fights_data['B_TD_pct'].str.strip('%').astype(float)

fights_data['R_SIG_STR_pct'] = fights_data['R_SIG_STR_pct']/100
fights_data['B_SIG_STR_pct'] = fights_data['B_SIG_STR_pct']/100
fights_data['R_TD_pct'] = fights_data['R_TD_pct']/100
fights_data['B_TD_pct'] = fights_data['B_TD_pct']/100

newfights_data = fights_data.drop(["R_TOTAL_STR.", "B_TOTAL_STR.", "R_TD", "B_TD"], axis=1)

#test code

significant = newfights_data[['R_HEAD','B_HEAD','R_BODY','B_BODY','R_LEG','B_LEG','R_CLINCH','B_CLINCH','R_GROUND','B_GROUND']]
sig_col = significant.columns
sig_df = pd.DataFrame()

for col in sig_col:
    sig_series = significant[col].str.split(' of ', expand = True)
    sig_df[col] = sig_series[0].astype(float)/sig_series[1].astype(float)

sig_df.fillna(0)

#end test code

temp_df1 = fights_data['R_TOTAL_STR.'].str.split(' of ', expand = True)
temp_df1['R_TOTAL_STR.'] = temp_df1[0].astype(float)/temp_df1[1].astype(float)

temp_df2 = fights_data['B_TOTAL_STR.'].str.split(' of ', expand = True)
temp_df2['B_TOTAL_STR.'] = temp_df2[0].astype(float)/temp_df2[1].astype(float)

temp_df3 = fights_data['R_TD'].str.split(' of ', expand = True)
temp_df3['R_TD'] = temp_df3[0].astype(float)/temp_df3[1].astype(float)

temp_df4 = fights_data['B_TD'].str.split(' of ', expand = True)
temp_df4['B_TD'] = temp_df3[0].astype(float)/temp_df3[1].astype(float)

newtemp_df1 = temp_df1.drop([0, 1], axis=1)
newtemp_df2 = temp_df2.drop([0, 1], axis=1)
newtemp_df3 = temp_df3.drop([0, 1], axis=1)
newtemp_df4 = temp_df4.drop([0, 1], axis=1)

df69 = newfights_data.join(newtemp_df1)
df70 = df69.join(newtemp_df2)
df71 = df70.join(newtemp_df3)
df72 = df71.join(newtemp_df4)

df72.head()

df72['new_column'] = np.where(df72['R_fighter'] == df72['Winner'], '0', '1')
df72.new_column.head()

df72 = df72.fillna(0)
df72 = df72.drop(['R_HEAD','B_HEAD','R_BODY','B_BODY','R_LEG','B_LEG','R_CLINCH','B_CLINCH','R_GROUND','B_GROUND'],axis=1)
df72 = df72

# Creating input feature
X = df72[['R_KD','B_KD','R_SIG_STR_pct', 'B_SIG_STR_pct', 'R_TOTAL_STR.', 'B_TOTAL_STR.',
                 'R_TD','B_TD','R_TD_pct','B_TD_pct','R_SUB_ATT','B_SUB_ATT','R_PASS','B_PASS','R_REV','B_REV']]

# Creating target variable
y = df72[['new_column']]

#Creating Train and Test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# #Normalizing the input features
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# X_train = sc_x.fit_transform(X_train)
# X_test = sc_x.transform(X_test)

# #Normalizing the target variable
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)
# y_test = sc_y.transform(y_test)

# Creating the Logistic regressor
classifier = LogisticRegression()

# fitting the training data to the Logistic Regressor
classifier.fit(X_train,y_train)

# Checking the model coefficients
classifier.coef_

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

classifier.fit(X_train, y_train)
test_results = classifier.predict(X_test)
print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")

predictions = classifier.predict(X_test)
print(f"First 10 Predictions:   {predictions[:10]}")
print(f"First 10 Actual labels: {y_test[:10].values.tolist()}")

accuracy = classifier.score(X_test, y_test)
print('The accuracy is: ' + str(accuracy *100) + '%')

pd.DataFrame({"Prediction": predictions, "Actual": y_test['new_column']}).reset_index(drop=True)

