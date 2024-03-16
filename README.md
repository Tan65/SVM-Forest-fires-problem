# SVM-Forest-fires-problem
Classify the Size_Categorie using SVM
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix

# Load the dataset
data = pd.read_csv('forestfires.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary Statistics
print(data.describe())

# Distribution Plots for Numerical Features
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Box Plots for Numerical Features
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data['size_category'], y=data[col])
    plt.title(f'Box Plot of {col} by size_category')
    plt.xlabel('size_category')
    plt.ylabel(col)
    plt.show()

# Categorical Plots
categorical_cols = ['month', 'day']
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x=col, hue='size_category')
    plt.title(f'Count Plot of {col} by size_category')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Pairwise Relationships using Scatter Plot
scatter_matrix(data[numerical_cols], figsize=(15, 15))
plt.suptitle('Pairwise Scatter Plot of Numerical Features', size=20)
plt.show()



# Separate the features (X) and the target variable (y)
X = data.drop('size_category', axis=1)
y = data['size_category']

# Perform label encoding on categorical columns
label_encoder = LabelEncoder()
for column in categorical_cols:
    X[column] = label_encoder.fit_transform(X[column])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM models with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel:")
    # Create an SVM model and train it
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Classification Report:\n', classification_rep)
