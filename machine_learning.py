'''
2. Anomaly Detection:

Idea: Identify anomalies or unusual behavior in the collected data that might indicate irregularities or significant events.
Code Example: Apply methods like Z-Score, Isolation Forest, or Autoencoders to detect anomalies.
'''

import pandas as pd

filepath = "C:/Users/hadab/OneDrive/Documents/synoptic project/b1.4.xlsx"
df = pd.read_excel(filepath)
df = df.fillna(0)
df = df.replace('-', 0)
df['RPM'] = df['RPM'].dt.strftime('%m-%Y')
df_transposed = df.T
df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed[1:]
df_transposed = df_transposed.reset_index(drop=True)

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load your BT data into a DataFrame
# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_transposed)
# Apply Isolation Forest for anomaly detection
clf = IsolationForest(contamination=0.05)
df_transposed['Is_Outlier'] = clf.fit_predict(normalized_data)

df_transposed.head()