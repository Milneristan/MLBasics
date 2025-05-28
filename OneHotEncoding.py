import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data={
    'CustomerID': ['1','2','3','4'],
    'Age': [25 ,np.nan, 30, 35],
    'Gender': ['Male','Female','Male','Female'],
    'Income': [np.nan, 25000, 30000, 35000],
    'City': ['New York', 'Los Angeles', np.nan, 'Chicago'],
    'Sub_status': ['NS', 'S', 'S', 'NS']
}


df = pd.DataFrame(data)
print(df, "\n")
print(df.isna(), "\n")
df_enc = pd.get_dummies(df, columns=['Gender', 'Sub_status'], drop_first=True)
print(f"One-Hot Encoded Data using Pandas:\n{df_pandas-encoded}\n")
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=encoder.get_feature_names_out(categorical_columns))
df_sklearn_encoded = pd.concat([df.drop(categorical_columns, axis=1), one_hot_df], axis=1)

print(f"One-Hot Encoded Data using Scikit-Learn:\n{df_sklearn_encoded}\n")
