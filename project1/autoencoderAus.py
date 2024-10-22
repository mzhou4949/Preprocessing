import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
sns.set_style("whitegrid")

np.random.seed(697)

# Import data
df = pd.read_csv('ausClean.csv')

# Convert label to binary (0 and 1)
df['label'] = (df['label'] == 1).astype(int)

# Scale variables using StandardScaler
scaler = StandardScaler()
columns_to_scale = [col for col in df.columns if col != 'label']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Split in 70% train, 15% dev, and 15% test set
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=1984, stratify=df['label'])
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=1984, stratify=temp_df['label'])

# Check distribution of labels
print(f"Train set default rate: {train_df.label.mean():.4f}")
print(f"Dev set default rate: {dev_df.label.mean():.4f}")
print(f"Test set default rate: {test_df.label.mean():.4f}")

# Define the final train and test sets
train_y = train_df.label
dev_y = dev_df.label
test_y = test_df.label

train_x = train_df.drop(['label'], axis=1)
dev_x = dev_df.drop(['label'], axis=1)
test_x = test_df.drop(['label'], axis=1)

train_x = np.array(train_x)
dev_x = np.array(dev_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
dev_y = np.array(dev_y)
test_y = np.array(test_y)

# Build the AutoEncoder
encoding_dim = 10

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

# Plot autoencoder loss
plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

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

# Plot model loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Encoded model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Predictions and visualizations
predictions_NN_prob = model.predict(encoded_test_x)
predictions_NN_prob = predictions_NN_prob[:,0]
predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0)

# Print accuracy
acc_NN = accuracy_score(test_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)

# Plot ROC curve
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = f'AUC = {roc_auc:.3f}')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

# Plot Confusion Matrix
cm = confusion_matrix(test_y, predictions_NN_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin=0.2)
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

#https://github.com/georsara1/Autoencoders-for-dimensionality-reduction/blob/master/autoencoder.py