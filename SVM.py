# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.externals import joblib
import joblib


# %%
# We have 11 features. To predict a quality of wine
wine = pd.read_csv('./winequality-red.csv', delimiter=";")

wine.columns  ##


# %%
# This is 5 first row of database

wine.head() ##

# %%
wine.describe()  ##


# %%
# Figure column quality by count of values.
# Worst score is 0, and best is 10, step is 1.
# The most number of wine have quality equal 6

plt.figure(figsize=(10, 6))
sns.countplot(wine["quality"], palette="muted")
wine["quality"].value_counts()


# %%
# Classification of score of quality into 3 groups: Bad, Medium and Good
quality = wine["quality"].values
category = []
for num in quality:
    if num <= 5:
        category.append("Bad")
    else:
        category.append("Good")


# %%
# Count number of wine which is MEDIUM category, BAD category and GOOD catergory
print([(i, category.count(i)) for i in set(category)])
plt.figure(figsize=(10, 6))
sns.countplot(category, palette="muted")


# %%
# Figure out the relate between each two feature in data.
# corr(x,y) = 0 if 2 feature absolute don't correlate to each other
# corr(x,y) > 0 if x increase then y also increase and vice versa.
# corr(x,y) < 0 if x increase then y decrease and vice versa.
# corr(x,y) = 1 or -1 if x and y absolutely correlation.
plt.figure(figsize=(12, 6))
sns.heatmap(wine.corr(), annot=True)


# %%
# Remove columns "quality" and add columns "category"
# X is all vector feature: all rows and all columns except last column
# y is output: all rows of last columns
category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([wine, category], axis=1)
data.drop(columns="quality", axis=1, inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# Figure out
data.columns


# %%
# Turn label of output y: "Good" = 1, "Bad" = 0.
labelencode_y = LabelEncoder()
y = labelencode_y.fit_transform(y)


# %%
# Separate cross-validation
skf = StratifiedShuffleSplit(n_splits=10,random_state=5)


# %%
# TRAINING WITH PARAMETER 1
classifier = SVC(C=0.1, gamma=0.1, kernel='rbf')
all_accuracies = cross_val_score(
    estimator=classifier,
    X=X,
    y=y,
    cv=skf,
    n_jobs=-1)
print(all_accuracies)
print("mean acc:",np.mean(all_accuracies))


# %%
# TRANING WITH PARAMETER 2
classifier = SVC(C=1, gamma=0.3, kernel='rbf')
all_accuracies = cross_val_score(
    estimator=classifier,
    X=X,
    y=y,
    cv=skf,
    n_jobs=-1)
print(all_accuracies)
print("mean acc:",np.mean(all_accuracies))


# Kernel SVM 'rbf'
# TUNING PARAMETER using GridSearchCV
# %%
pipe_svm = Pipeline([('clf', svm.SVC())])
params_C=np.geomspace(pow(2,-5),pow(2,15),num=21)
params_g=np.geomspace(pow(2,-15),pow(2,3),num=19)
grid_params = dict(clf__C=params_C,
                   clf__gamma=params_g,
                   clf__kernel=['rbf'])
gs_svm = GridSearchCV(estimator=pipe_svm,
                      param_grid=grid_params,
                      scoring='accuracy',
                      cv=skf,
                      n_jobs=-1)
gs_svm.fit(X, y)
# %%
print(gs_svm.best_score_)
print(gs_svm.best_params_)
