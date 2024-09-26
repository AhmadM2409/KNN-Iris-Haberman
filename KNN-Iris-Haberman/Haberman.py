import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Load the Haberman dataset
haberman_df = pd.read_csv('haberman.csv')  # Ensure this file is in your project directory

# Prepare features and labels
X = haberman_df.iloc[:, :-1].values  # Features: age, year, nodes
y = haberman_df.iloc[:, -1].values  # Labels: survival (1 or 2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN instance and fit to the data
knn = KNN(k=5)  # You can adjust k as needed
knn.fit(X_train, y_train)

# Predict and evaluate
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Haberman Dataset Accuracy: {accuracy:.2f}')
