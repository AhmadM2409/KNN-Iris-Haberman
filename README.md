KNN Classifier for Iris Dataset
Overview
This project implements a K-Nearest Neighbors (KNN) classifier using the Iris dataset, a classic dataset in the field of machine learning. The KNN algorithm is a simple yet powerful supervised learning method used for classification tasks. The project includes advanced features such as data preprocessing, hyperparameter tuning, and cross-validation to enhance model performance.

Dataset
The Iris dataset consists of 150 samples of iris flowers, categorized into three species: Iris Setosa, Iris Versicolor, and Iris Virginica. Each sample has four features:

Sepal Length
Sepal Width
Petal Length
Petal Width


Features
Data Preprocessing: The features are standardized using StandardScaler to improve model performance.
Hyperparameter Tuning: Utilizes Grid Search to determine the optimal number of neighbors (k) for the KNN algorithm.
Cross-Validation: Implements k-fold cross-validation to evaluate the model's performance, providing a robust assessment of its predictive capability.
Performance Metrics: The project includes detailed classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix to assess model performance.


Getting Started
Prerequisites
To run this project, ensure you have the following Python packages installed:

numpy
pandas
scikit-learn
You can install the required packages using pip:

bash
Copy code
pip install numpy pandas scikit-learn
Running the Code
Clone the repository:
bash
Copy code
git clone https://github.com/AhmadM2409/KNN-Iris-Haberman.git
Navigate to the project directory:
bash
Copy code
cd KNN-Iris-Haberman
Run the KNN implementation for the Iris dataset:
bash
Copy code
python "KNN iris.py"
Results
Upon running the code, you will see the model's performance metrics, including accuracy and a detailed classification report.

