import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv('data/cybersecurity_attacks.csv')

# Convert categorical features to numeric and ensure numerical features are floats
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = LabelEncoder().fit_transform(df[column])
    elif df[column].dtype in ['int64', 'int32']:
        df[column] = df[column].astype('float32')

# Separate features and target
X = df.drop('Attack Type', axis=1)  # Assuming 'attack_type' is the target column
y = df['Attack Type']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier with size control parameters
model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_leaf_nodes=50,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report with correct target names
try:
    target_names = [str(class_name) for class_name in label_encoder.classes_]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print('Classification Report:')
    print(report)
except Exception as e:
    print(f"Error in generating classification report: {e}")

# Convert feature names to strings
feature_names = [str(feature) for feature in X.columns]

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=feature_names, class_names=target_names, rounded=True)
plt.title('Decision Tree')
plt.show()
