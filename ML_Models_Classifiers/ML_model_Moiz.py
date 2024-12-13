"""Loading in packages"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

#Standardizing features we wont need it since Isha is already doing this but I 
#put it here just in case
def standardize_features(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized
#classifying data into features and targets
def features_and_target(data):
    X= data.iloc[:, :-1].values
    y= data.iloc[:, -1].values
    return (X,y)
# splitting the data
def data_split(X, y, v):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=v, random_state=24)
    return (X_train, X_test, y_train, y_test)
#fitting the model
def fit_model(type_class, x,y):
    model = type_class.fit(x, y)
    return (model)
#Testing performance
def model_Testing(model,x,y,):
    perf = model.score(x,y)
    print(f"Model Performance Score: {perf}")
    return(perf)
#Accuracy 
def accuracy_testing(model,x,y):
    y_pred = model.predict(x)
    accu_test= accuracy_score(y, y_pred)
    print(f'The accuracy of this model is: {accu_test}')
    return accu_test

#will need to use the line below if using KNN ML 
#knn = KNeighborsClassifier(n_neighbors=3)

#for testing purposes
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

#the final show
X,y=features_and_target(data)
X = standardize_features(X)
X_train, X_test, y_train, y_test = data_split(X, y, 0.2)
model = fit_model(LogisticRegression(max_iter=200), X_train,y_train)
performance = model_Testing(model,X_test,y_test)
accuracy = accuracy_testing(model, X_test,y_test)

"""Visualization of the results"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Confusion matrix
#Prediction
y_pred = model.predict(X_test)
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
#visualization
# Create a Confusion Matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Heatmap
correlation_matrix = data.corr()

#Plotting the data
plt.figure(figsize=(10, 8))  # Adjust size as needed
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap")
plt.show()