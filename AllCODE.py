#Created by the one and only
#  ____            _           _   ____              _             
# / ___|  ___ _ __(_) ___  ___| | |  _ \ __ _ _ __  (_) __ _ _ __  
# \___ \ / _ \ '__| |/ _ \/ _ \ | | |_) / _` | '_ \ | |/ _` | '_ \ 
#  ___) |  __/ |  | |  __/  __/ | |  _ < (_| | | | || | (_| | | | |
# |____/ \___|_| _/ |\___|\___|_| |_| \_\__,_|_| |_|/ |\__,_|_| |_|
#               |__/                              |__/             


# 1
# Import suitable Python Statistics Libraries and measure the central
# tendency of the given data using Mean, Weighted mean, harmonic mean,
# geometric mean, Median, and Mode using suitable method.

from numpy import mean, median, average
from pandas import read_csv
from scipy.stats import hmean, gmean
from statistics import mode

df = read_csv('diabetes.csv')
data = df['Insulin'].tolist()

print("Mean:", mean(data))
print("Weighted Mean:", average(data))
print("Harmonic Mean:", hmean(data))
print("Geometric Mean:", gmean(data))
print("Median:", median(data))
print("Mode:", mode(data))

# 2
# Import suitable Python Statistics Libraries and measure the variability of
# the given data using variance, standard deviation, skewness, quartiles,
# percentiles and range using suitable method

import pandas as pd

df = pd.read_csv('diabetes.csv')
data = df['Insulin']

print("Variance:", data.var())
print("Standard Deviation:", data.std())
print("Skewness:", data.skew())
print("Quartiles:", data.quantile([0.25, 0.5, 0.75]))
print("Percentiles:", data.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
print("Range:", data.max() - data.min())

# 3
# Import suitable Python Statistics Libraries and measure the correlation
# and covariance between given pairs of data using suitable method

import pandas as pd

df = pd.read_csv('diabetes.csv')

print("Correlation:", df.corr())
print("Covariance:", df.cov())

# 4
# Import suitable Python Libraries, explore the given dataset and visualize
# using boxplot, histogram, piechart, barchart and heatmaps

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')

# boxplot
fig = plt.figure(figsize=(10, 10))
plt.boxplot(df['BloodPressure'])
plt.show()

# histogram
fig = plt.figure(figsize=(10, 10))
plt.hist(df['BloodPressure'])
plt.show()

# piechart
fig = plt.figure(figsize=(10, 10))
plt.pie(df['BloodPressure'])
plt.show()

# barchart
fig = plt.figure(figsize=(10, 10))
plt.bar(df['BloodPressure'], width=0.5, height=0.5)
plt.show()

# heatmap
fig = plt.figure(figsize=(10, 10))
plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
plt.show()

# 5
# Implement Python code to implement data pre-processing
# functionalities.
# a. Handling Missing values – Delete missing entries, Fill with Null/0, Fill
# with Mean
# b. Data Smoothing using Binning – Bin by means and boundaries.
# c. Find out Co-variance and Correlation between 2 attributes using
# python.

import pandas as pd
import numpy as np

# data
dict = {'scores': [100, 90, np.nan, 95, 66, 5, 3, np.nan, 54, np.nan, 35, 48, 3,
                   np.nan, 43, 3, 67, 35, np.nan, np.nan, np.nan, 99, 77, 42, np.nan, 34, np.nan]}

df = pd.DataFrame(dict)
# handling missing values
df['scores'].fillna(df['scores'].mean(), inplace=True)
# print(df)

# binning by means
print(pd.cut(df['scores'], bins=3, labels=["Low", "Medium", "High"]))

# binning by boundries
print(pd.cut(df['scores'], bins=[
      min(df['scores']), max(df['scores'])], labels=["bin"]))

# 6
# Implement Python code to apply Apriori algorithm to mine association
# rules for the given dataset. Use suitable packages.
# a.Use Store_dataset attached.
# b.List all the rules with Min_support = 0.06, Confidence = 0.5
# Lift = 3 of min_length =3.

import pandas as pd
from apyori import apriori

df = pd.read_csv('store_data_Association.csv',header=None)

data = []
for i in range(0,500):
    data.append([str(df.values[i,j]) for j in range(0,20)])

association_results = list(apriori(data, min_support=0.006, min_confidence=0.5, min_lift=3, min_length=3))

for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule :"+ str(items[0]) + "->" + str(items[1]))
    print("Support : {}".format(item[1]))
    print("Confidence : {}".format(item[2][0][2]))
    print("List : {}".format(item[2][0][3]))
    print("\n-------------------------------------------------\n")

# 7
# Write a python code to identify frequent 1-Itemset from the given dataset
# with min-support=50

from collections import defaultdict
import pandas as pd

df=pd.read_csv("./store_data_Association.csv",header=None)

data = [[str(df.values[i,j]) for j in range(0,20)] for i in range(7501)]

support_count=defaultdict(int)

for i in range(0,7501):
    for j in range(0,20):
        support_count[data[i][j]]+=1

print([k for k, v in support_count.items() if v >= 50])

# 8
# For the given dataset, apply naïve bayes classifier to build model and
# predict class label for the test set (Overcast, Mild), (Sunny, Cool), (Rainy,
# Hot) using python scikit.

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data = {'weather': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'], 'temp': ['hot', 'hot', 'hot',
                                                                                                                                                                        'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'], 'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']}

le = LabelEncoder()
le1 = LabelEncoder()
df = pd.DataFrame.from_dict(data)

clel = le.fit(df.weather)
cle2 = le1.fit(df.temp)

df.weather = clel.transform(df.weather)
df.temp = cle2.transform(df.temp)

features = df.loc[:, df.columns != 'play']
print(features)
cls_label = df['play']

gnb = GaussianNB()
gnb.fit(features, cls_label)
test_data = {'weather': ['overcast', 'sunny',
                         'rainy'], 'temp': ['mild', 'cool', 'hot']}
test = pd.DataFrame.from_dict(test_data)

test.weather = clel.transform(test.weather)
test.temp = cle2.transform(test. temp)
print(gnb.predict(test))

# 9
# Use the given “iris.csv” dataset. Apply Gaussian Naïve Bayes Model for
# prediction. Print the classification report and confusion matrix. Write a
# short inference from the result.

from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('Iris.csv')

le_weather = preprocessing.LabelEncoder()
df['Species'] = le_weather.fit_transform(df['Species'])
print(le_weather.classes_)


X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

baseline = GaussianNB()
baseline.fit(X_train, y_train)
print('model has been trained... im ready to make predictions')


y_pred = baseline.predict(X_test)
confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=[
      'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
# 10
# Use the given Pima Indian Diabetes dataset.Load the dataset and split
# dataset as features and target variable. Then split the dataset into
# training and test dataset. Apply decision tree classifier, calculate
# accuracy and visualize the decision tree

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
print(f'the number of null values : {df.isna().sum().sum()}')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2022)

model = DecisionTreeClassifier() 
model.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = model.predict(X_test)

print(f'The accuracy score is : {accuracy_score(y_test, y_pred)}')

fig = plt.figure(figsize=(25, 20))
tree.plot_tree(model,
               feature_names=X.columns,
               class_names=['0', '1'],
               filled=True)
plt.show()

# 11
# For the given dataset balance-scale.data, load the dataset and set the
# target and feature variables. Split the dataset into training and test
# dataset. Perform the following task:
# I. Build decision tree classifier with Entropy criteria
# II. Perform Prediction for test dataset using Entropy and print the
# results in the form of confusion matrix, accuracy and
# classification report.

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

df = pd.read_csv('balance-scale.csv')
print(f'the number of null values : {df.isna().sum().sum()}')

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2022)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = model.predict(X_test)

print(f'The accuracy score is : {accuracy_score(y_test, y_pred)}')
print("Classification Report: \n", classification_report(
    y_test, y_pred, target_names=['L', 'B', 'R']))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# 12
# For the given 2- dimensional dataset, apply K-means clustering to cluster
# the dataset into 2 clusters. Plot the graph and display the result.

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]}
  
df = DataFrame(Data,columns=['x','y'])
  
kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_
print("Centroids: \n",centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

# 13
# Create a random dataset of 30 elements with x and y variables using
# random function between 20 to 60 integers for x and 50 to 90 integers
# for y. Apply K- means clustering to cluster the data into 2 clusters.

import numpy as np
from matplotlib import pyplot as plt
import cv2

X = np.random.randint(20, 60, (30, 2))
Y = np.random.randint(50, 90, (30, 2))
Z = np.vstack((X,Y))

# convert to np.float32
Z = np.float32(Z)
  
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center = cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  
# Now separate the data
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
  
# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Test Data'),plt.ylabel('Z samples')
plt.show()