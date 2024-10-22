import pandas  as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# adult has 48842 data samples in total with 23.93% as positive (prediction of income as >50K) and the rest 76.07% as negative (prediction of income as <=50K) 
# where there are 14 attributes that are mixed with continuous and categorical variables
#age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income

#dfadult1 = pd.read_csv('adult.csv', header=None,skipinitialspace=True,na_values=["?"])
dfadult = pd.read_csv('adultTest.csv', header=None,skipinitialspace=True,na_values=["?"])

#dfadult = pd.concat([dfadult1, dfadult2], axis=0)
dfadult.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]
print(dfadult.info())
print(dfadult.duplicated().sum())

#Check for missing values
print(dfadult.isnull().sum())
# Identify categorical columns (columns with dtype 'object')
categorical_cols = dfadult.select_dtypes(include=['object']).columns

# For each categorical column, fill NaN values with the mode (most frequent value)
for col in categorical_cols:
    mode_value = dfadult[col].mode()[0]  # Get the mode of the column
    dfadult[col] = dfadult[col].fillna(mode_value)  # Replace NaN with mode

print(dfadult.isnull().sum())

df_no_duplicates = dfadult.drop_duplicates()

print(df_no_duplicates.duplicated().sum())
continuous_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

# 2. Handle categorical attributes with one-hot encoding (or label encoding)
#One-hot encoding better than label encoding becuase our categorical data doesn't have heirarchy(no order) 
categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                       'race', 'sex', 'native_country']

# Apply one-hot encoding for all categorical columns
#df = pd.get_dummies(df_no_duplicates, columns=categorical_columns)
label_encoder = LabelEncoder()
df = df_no_duplicates
for col in categorical_columns:
    df.loc[:,col] = label_encoder.fit_transform(df[col])

# 3. Handle the target variable (income)
df.loc[:,'income'] = label_encoder.fit_transform(df['income'])  # <=50K becomes 0, >50K becomes 1
df_no_duplicates.to_csv('adultCleanedTest.csv',index = False)

pd.set_option('display.max_columns', None)  # Show all columns

dfPCA = df.drop('income',axis =1)
scaler = StandardScaler() #different matrix
print(dfPCA)
X_standardized = scaler.fit_transform(dfPCA)
cov_matrix = np.cov(X_standardized.T)

print("\nCovariance Matrix Shape:", cov_matrix.shape) 

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort the eigenvalues and corresponding eigenvectors in descending order
 # Indices for sorting in descending order
idx = eigenvalues.argsort()[::-1] 
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues:\n", eigenvalues)
# Calculate the cumulative explained variance
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Plot cumulative explained variance to decide the number of components

plt.plot(cumulative_explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Principal Components')
plt.show()

print("Component | Explained Variance Ratio | Cumulative Explained Variance")
for i, (evr, cev) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance), 1):
    print(f"PC{i:02d}       | {evr:.6f}                | {cev:.6f}")

# Example: Select the number of components that explain 95% of the variance
n_components = np.argmax(cumulative_explained_variance >= 0.80) + 1
print(f"\nNumber of components to retain 95% variance: {n_components}")

# Project the data onto the selected principal components
X_pca_95 = X_standardized.dot(eigenvectors[:, :n_components])

# Convert the PCA-reduced data to a DataFrame for easier analysis
df_pca_95 = pd.DataFrame(X_pca_95, columns=[f'PC{i+1}' for i in range(n_components)])
print(f"\nShape of the data after PCA (reduced to {n_components} components):", df_pca_95.shape)

