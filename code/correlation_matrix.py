import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define chunk size
chunk_size = 10000

# Load and process data in chunks
chunk_list = []  # append each chunk df here

for chunk in pd.read_csv('data/cybersecurity_attacks.csv', chunksize=chunk_size):
    # Convert categorical features to numeric
    for column in chunk.columns:
        if chunk[column].dtype == 'object':
            chunk[column] = LabelEncoder().fit_transform(chunk[column])
        elif chunk[column].dtype == 'float64':
            chunk[column] = chunk[column].astype('float32')
        elif chunk[column].dtype == 'int64':
            chunk[column] = chunk[column].astype('int32')
    chunk_list.append(chunk)

# Concatenate all chunks into a single dataframe
df = pd.concat(chunk_list)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of the Dataset')
plt.show()
