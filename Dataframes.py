##decorator function

import pandas as pd
import numpy as np
data={
    'CustomerID': ['1','2','3','4'],
    'Age': [25 ,np.nan, 30, 35],
    'Gender': ['Male','Female','Male','Female'],
    'Income': [np.nan, 25000, 30000, 35000],
    'City': ['New York', 'Los Angeles', np.nan, 'Chicago'],
    'Sub_status': ['NS', 'S', 'S', np.nan]
}


df = pd.DataFrame(data)
df_dropped = df.dropna()
print(df_dropped)
df_filled = df.fillna(value={'CustomerID': 'Unknown',
                             'Age': df['Age'].mean(),
                             'Gender': 'Unknown',
                             'Income': df['Income'].mean(),
                             'City': 'Unknown',
                             'Sub_status': 'Unknown'})
                      
print(df, "\n")
print(df.isna(), "\n")
print(df_filled)



