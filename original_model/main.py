import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform Linear Discriminant Analysis (LDA)
lda = LDA(n_components=1)  # n_components=1 for 2-class separation
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Visualize the separation using LDA-transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_train_lda[y_train == 0], np.zeros_like(X_train_lda[y_train == 0]), color='red', alpha=0.8, label='Class 0')
plt.scatter(X_train_lda[y_train == 1], np.zeros_like(X_train_lda[y_train == 1]), color='blue', alpha=0.8, label='Class 1')
plt.xlabel('LDA Component 1')
plt.title('LDA Separation of Two Classes')
plt.legend()
plt.show()

# Evaluate classification accuracy
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
