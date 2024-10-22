import pandas  as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv('australian.csv', header=None, sep = r"\s",engine='python', na_values=["?"])

df['label'] = df[0]  # Save the first column as a label
df.drop(0, axis=1, inplace=True)  # Drop the first column

df = df.astype(str)

# Step 2: Split key-value pairs into separate values
for col in df.columns[:-1]:
    df[col] = df[col].str.split(":").str[1]  # Extract the value after the ":"

# Convert the values to numeric
df = df.apply(pd.to_numeric)

print(df.isnull().sum()) #no missing data

print(df.duplicated().sum()) #no duplicates

df.to_csv('ausClean.csv',index = False)

#no missing data and duplicates
X = df.drop(columns=['label'])  # Remove label column for PCA

# Step 3: Standardize the data using StandardScaler
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X) 

cov_matrix = np.cov(X_standardized, rowvar=False)

# Step 5: Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 6: Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 7: Calculate explained variance ratio for each principal component
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

# Step 8: Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Print explained variance for each component
for i, (evr, cev) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance), 1):
    print(f"Principal Component {i}: Explained Variance = {evr:.4f}, Cumulative Explained Variance = {cev:.4f}")

# Step 9: Plot the explained variance and cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Explained Variance')
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', label='Cumulative Explained Variance', color='r')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Explained')
plt.legend()
plt.grid(True)
plt.show()