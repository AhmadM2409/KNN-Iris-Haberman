import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def set_k(self, k):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def perform_cross_validation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    print(f'Cross-validation scores: {scores}')
    print(f'Mean score: {scores.mean()}')

def tune_hyperparameters(X_train, y_train):
    param_grid = {'n_neighbors': np.arange(1, 21)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(f'Best k: {grid_search.best_params_["n_neighbors"]}')
    return grid_search.best_params_["n_neighbors"]

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize KNN classifier
    knn = KNNClassifier()

    # Hyperparameter tuning
    best_k = tune_hyperparameters(X_train, y_train)
    knn.set_k(best_k)

    # Fit the model
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # Performance metrics
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Perform cross-validation
    perform_cross_validation(knn.model, X_train, y_train)

if __name__ == "__main__":
    main()
