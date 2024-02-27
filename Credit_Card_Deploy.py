# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder
# Loading the dataset
df = pd.read_csv('creditcard.csv')

# Renaming DiabetesPedigreeFunction as DPF
#df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']] = df_copy[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Time'].fillna(df_copy['Time'].mean(), inplace=True)
df_copy['V1'].fillna(df_copy['V1'].mean(), inplace=True)
df_copy['V2'].fillna(df_copy['V2'].mean(), inplace=True)
df_copy['V3'].fillna(df_copy['V3'].mean(), inplace=True)
df_copy['V4'].fillna(df_copy['V4'].mean(), inplace=True)
df_copy['V5'].fillna(df_copy['V5'].mean(), inplace=True)
df_copy['V6'].fillna(df_copy['V6'].mean(), inplace=True)
df_copy['V7'].fillna(df_copy['V7'].mean(), inplace=True)
df_copy['V8'].fillna(df_copy['V8'].mean(), inplace=True)
df_copy['V9'].fillna(df_copy['V9'].mean(), inplace=True)
df_copy['V10'].fillna(df_copy['V10'].mean(), inplace=True)
df_copy['V11'].fillna(df_copy['V11'].mean(), inplace=True)
df_copy['V12'].fillna(df_copy['V12'].mean(), inplace=True)
df_copy['V13'].fillna(df_copy['V13'].mean(), inplace=True)
df_copy['V14'].fillna(df_copy['V14'].mean(), inplace=True)
df_copy['V15'].fillna(df_copy['V15'].mean(), inplace=True)
df_copy['V16'].fillna(df_copy['V16'].mean(), inplace=True)
df_copy['V17'].fillna(df_copy['V17'].mean(), inplace=True)
df_copy['V18'].fillna(df_copy['V18'].mean(), inplace=True)
df_copy['V19'].fillna(df_copy['V19'].mean(), inplace=True)
df_copy['V20'].fillna(df_copy['V20'].mean(), inplace=True)
df_copy['V21'].fillna(df_copy['V21'].mean(), inplace=True)
df_copy['V22'].fillna(df_copy['V22'].mean(), inplace=True)
df_copy['V23'].fillna(df_copy['V23'].mean(), inplace=True)
df_copy['V24'].fillna(df_copy['V24'].mean(), inplace=True)
df_copy['V26'].fillna(df_copy['V26'].mean(), inplace=True)
df_copy['V27'].fillna(df_copy['V27'].mean(), inplace=True)
df_copy['V28'].fillna(df_copy['V28'].median(), inplace=True)
df_copy['Amount'].fillna(df_copy['Amount'].median(), inplace=True)


# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='Class')
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'CreditCard-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))