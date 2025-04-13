import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.dropna()

    le = LabelEncoder()
    df.loc[:,'Churn'] = le.fit_transform(df['Churn'])

    for col in df.select_dtypes(include = ['object']).columns:
        if col != 'Customer ID':
            df[col] = le.fit_transform(df[col])

    x = df.drop(['customerID', 'Churn'], axis = 1)
    y = df['Churn']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return train_test_split(x_scaled, y, test_size= 0.2, random_state = 42)