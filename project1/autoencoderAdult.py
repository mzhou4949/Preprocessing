import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
sns.set_style("whitegrid")

np.random.seed(697)

# Import data
df = pd.read_csv('adultCleaned.csv')
dftest = pd.read_csv('adultCleanedTest.csv')

print("Unique values in 'income' column:", df['income'].unique())
print("Distribution of 'income':")
print(df['income'].value_counts(normalize=True))

# Define numeric and categorical columns
numeric_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

# Scale numeric variables using StandardScaler
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
dftest[numeric_columns] = scaler.fit_transform(dftest[numeric_columns])

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
dftest = pd.get_dummies(dftest, columns=categorical_columns, drop_first=True)

# Split in 70% train, 15% dev, and 15% test set
# train_df, temp_df = train_test_split(df, test_size=0.3, random_state=1984, stratify=df['income'])
# dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=1984, stratify=temp_df['income'])
# Align columns between the training and test set to ensure the same structure
df, test_df = df.align(dftest, join='left', axis=1, fill_value=0)

# Split the training data into 85% train and 15% dev set
train_df, dev_df = train_test_split(df, test_size=0.15, random_state=1984, stratify=df['income'])
# Check distribution of income
print(f"Train set income >50K rate: {train_df.income.mean():.4f}")
print(f"Dev set income >50K rate: {dev_df.income.mean():.4f}")
print(f"Test set income >50K rate: {test_df.income.mean():.4f}")

# Define the final train and test sets
train_y = train_df.income.values
dev_y = dev_df.income.values
test_y = test_df.income.values

train_x = train_df.drop(['income'], axis=1).values
dev_x = dev_df.drop(['income'], axis=1).values
test_x = test_df.drop(['income'], axis=1).values

print("Unique values in train_y:", np.unique(train_y))
print("Unique values in dev_y:", np.unique(dev_y))
print("Unique values in test_y:", np.unique(test_y))

# Ensure all data is float32
train_x = train_x.astype('float32')
dev_x = dev_x.astype('float32')
test_x = test_x.astype('float32')

# Build the AutoEncoder
encoding_dim = 8

input_data = Input(shape=(train_x.shape[1],))
encoded = Dense(16, activation='relu')(input_data)
encoded = BatchNormalization()(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(train_x.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

hist_auto = autoencoder.fit(train_x, train_x,
                            epochs=100,
                            batch_size=64,
                            shuffle=True,
                            validation_data=(dev_x, dev_x))

# Create encoder model
encoder = Model(input_data, encoded)

# Encode data set
encoded_train_x = encoder.predict(train_x)
encoded_dev_x = encoder.predict(dev_x)
encoded_test_x = encoder.predict(test_x)

# Build new model using encoded data
model = Sequential()
model.add(Dense(32, input_dim=encoding_dim, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(encoded_train_x, train_y, 
                    validation_data=(encoded_dev_x, dev_y),
                    epochs=100, 
                    batch_size=32)

# Predictions and visualizations
predictions_NN_prob = model.predict(encoded_test_x)
predictions_NN_prob = predictions_NN_prob[:,0]
predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0)

print("Unique values in predictions:", np.unique(predictions_NN_01))
print("Distribution of predictions:")
print(pd.Series(predictions_NN_01).value_counts(normalize=True))

# Print accuracy
acc_NN = accuracy_score(test_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)

# Plot Confusion Matrix
cm = confusion_matrix(test_y, predictions_NN_01)
labels = ['Income <= 50K', 'Income > 50K']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin=0.2)
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

#https://github.com/georsara1/Autoencoders-for-dimensionality-reduction/blob/master/autoencoder.py
