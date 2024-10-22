import pandas  as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('emg.csv', header=None, sep = r"\s",engine='python', na_values=["?"])

df['label'] = df[0]  # Save the first column as a label
df.drop(0, axis=1, inplace=True)  # Drop the first column


df = df.astype(str)

# Step 2: Split key-value pairs into separate values
for col in df.columns[:-1]:
    df[col] = df[col].str.split(":").str[1]  # Extract the value after the ":"

# Convert the values to numeric
df = df.apply(pd.to_numeric)

print(df.isnull().sum()) #no missing data

print(df.duplicated().sum())
df_no_duplicates = df.drop_duplicates()
print(df_no_duplicates.duplicated().sum())

df_no_duplicates.to_csv('adultCleanedEmg.csv',index = False)

#duplicates removed and no missing data

#remove the first column

dfDCT= df_no_duplicates.drop(columns=['label'])  # Remove label column for DCT

num_rows, num_features = dfDCT.shape

# Create an empty array to store the DCT coefficients for all rows
dct_all_rows = np.zeros((num_rows, num_features))  # Now it's defined

# Loop over all rows in the DataFrame to apply DCT to each row
for i in range(num_rows):
    row = dfDCT.iloc[i].values  # Get the i-th row as an array
    N = len(row)  # Length of the row (number of features)
    dct_row = np.zeros(N)  # Empty array to store DCT for the current row

    # DCT-II formula for each row
    for k in range(N):
        sum_value = 0
        for n in range(N):
            sum_value += row[n] * np.cos(np.pi * (n + 0.5) * k / N)
        dct_row[k] = sum_value

    # Store the DCT coefficients for the row in the dct_all_rows array
    dct_all_rows[i] = dct_row


# Compute the squared DCT coefficients (variance captured by each component)
squared_dct_all = dct_all_rows ** 2

# Compute the total variance for each row
total_variance = np.sum(squared_dct_all, axis=1)

# Compute the explained variance ratio for each DCT component
explained_variance_ratio = squared_dct_all / total_variance[:, np.newaxis]

# Compute the cumulative explained variance for the first row (as an example)
cumulative_explained_variance = np.cumsum(explained_variance_ratio[0])

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of DCT Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by DCT Components (First Row)')
plt.grid(True)
plt.show()

# Find the number of components required to capture 95% of the variance
k = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f"Number of DCT components to retain 95% of the variance: {k}")

# Retain only the first k components
dct_reduced = dct_all_rows[:, :k]

# Print the shape of the reduced data
print(f"Shape of the data after selecting the first {k} DCT components: {dct_reduced.shape}")
