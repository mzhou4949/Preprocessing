import pandas  as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# adult has 48842 data samples in total with 23.93% as positive (prediction of income as >50K) and the rest 76.07% as negative (prediction of income as <=50K) 
# where there are 14 attributes that are mixed with continuous and categorical variables
#age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income

dfadult1 = pd.read_csv('adult.csv', header=None,skipinitialspace=True,na_values=["?"])
dfadult2 = pd.read_csv('adultTest.csv', header=None,skipinitialspace=True,na_values=["?"])

dfadult = pd.concat([dfadult1, dfadult2], axis=0)

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
pd.set_option('display.max_columns', None)  # Show all columns

dfDCT = df.drop('income',axis =1)


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