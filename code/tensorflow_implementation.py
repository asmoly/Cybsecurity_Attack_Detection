# Dataset: https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks?resource=download

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, AdamW
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/cybersecurity_attacks.csv')

# Convert categorical features to numeric
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

print(df.head())

# Preprocess the data
# X = df.drop(['Attack Type', 'Action Taken', 'Severity Level',
# 'User Information', 'Device Information', 'Network Segment',
# 'Proxy Information', 'Firewall Logs', 'Log Source'], axis=1)  # Assuming 'attack_type' is the target column
# y = df['Attack Type']
X = df.drop(['Attack Type', 'Timestamp', 'Source IP Address', 'Payload Data', 'Alerts/Warnings',
             'Action Taken', 'Severity Level', 'User Information', 'Device Information',
             'Network Segment', 'Geo-location Data', 'Proxy Information', 'Firewall Logs', 'IDS/IPS Alerts',
             'Log Source'], axis=1)  # Assuming 'attack_type' is the target column
y = df['Attack Type']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

optimizer = AdamW(learning_rate=0.1)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predicting the type of attack
def predict_attack(data):
    prediction = model.predict(data)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Example usage
example_data = np.array([X_test[0]])  # Replace this with your own data
print(predict_attack(example_data))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
