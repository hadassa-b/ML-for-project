# import data
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

## dealing with negative values in series
'''
1. Shift and Add Constant:
One common approach is to shift the data by a constant to make it positive. For example, you can add the absolute minimum value plus a small constant to the entire series.
'''
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Create sample data with negative values
data = [-2, 1, -4, 3, -1, 0, -3, 2]

# Shift and add constant to make data positive
min_value = abs(min(data)) + 0.1
data_shifted = [x + min_value for x in data]

# Perform seasonal decomposition
result = seasonal_decompose(data_shifted, model='additive')
result.plot()
plt.show()

'''
2. Use Log Transformation:
Another approach is to apply a logarithmic transformation to the data. This can help stabilize the variance and make it more suitable for decomposition.
'''
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Create sample data with negative values
data = [-2, 1, -4, 3, -1, 0, -3, 2]

# Apply log transformation
data_log = np.log(np.array(data) + 1)  # Adding 1 to handle zero values

# Perform seasonal decomposition
result = seasonal_decompose(data_log, model='additive')
result.plot()
plt.show()


'''
1. Time Series Analysis and Forecasting:

Idea: Analyze historical trends and patterns in the collected data to forecast future values.
Code Example: Use time series libraries like pandas and statsmodels to perform time series decomposition, trend analysis, and forecasting.

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load your BT data into a DataFrame
# Perform time series decomposition
result = seasonal_decompose(df['B8ZD'], model='multiplicative')
result.plot()
plt.show()
'''

'''
2. Anomaly Detection:

Idea: Identify anomalies or unusual behavior in the collected data that might indicate irregularities or significant events.
Code Example: Apply methods like Z-Score, Isolation Forest, or Autoencoders to detect anomalies.


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
'''

'''
3. Correlation Analysis:

Idea: Investigate relationships and correlations between different items collected by UK MFIs, which can inform the Bank's understanding of the financial sector.
Code Example: Calculate correlations between variables and visualize them using libraries like seaborn.

import seaborn as sns
import matplotlib.pyplot as plt

# Load your BT data into a DataFrame
# Calculate correlation matrix
correlation_matrix = df.corr()
# Visualize correlations using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()
'''

'''
4. Clustering and Segmentation:

Idea: Group similar UK MFIs based on their collected data characteristics to gain insights into different segments of the financial sector.
Code Example: Use clustering algorithms like K-Means or DBSCAN.

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your BT data into a DataFrame
# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df)
# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(normalized_data)
'''

'''
5. Time Series Decomposition:

Idea: Break down the time series data into trend, seasonal, and residual components to understand underlying patterns.
Code Example: Use time series libraries like statsmodels for decomposition.

'''
from statsmodels.tsa.seasonal import seasonal_decompose

# Load your BT data into a DataFrame
# Perform time series decomposition
result = seasonal_decompose(df['B8ZD'], model='multiplicative')
trend = result.trend
seasonal = result.seasonal
residual = result.resid
