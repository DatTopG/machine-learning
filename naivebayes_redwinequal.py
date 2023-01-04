# Import library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

# The datasets of red wine have 11 features 
data =pd.read_csv("winequality-red.csv",sep=";")

data

# The describe about datasets 
data.describe()

# Checking for missing/null values 
data.isna().sum()

#checking the distribution of quality variable
 # The number of values figure the column quality, in datasets the worst quality score is 3 and the best quality score is 8
plt.figure(figsize=(10, 6))
sns.countplot(data["quality"], palette="muted")

# Obtaing Correlation Matrix for the dataset
# corr(x,y) = 0 if 2 feature absolute don't correlate to each other
# corr(x,y) > 0 if x increase then y also increase and vice versa.
# corr(x,y) < 0 if x increase then y decrease and vice versa.
# corr(x,y) = 1 or -1 if x and y absolutely correlation.
fig, ax =plt.subplots(figsize=(10,10))
dataplot = sns.heatmap (data.corr(), annot = True, ax=ax)

# Classify the score of quality into 2 groups: Bad and Good
# Quality of wine is good if the score is higher than 6 and bad elsewise
category=[]
for i in data["quality"]:
  if i<=5:
    category.append("bad")
  if i>=6:
    category.append("good")
  
cate=pd.DataFrame(data=category,columns=["category"])
data=pd.concat([data,cate],axis =1)
data.drop(columns="quality",axis=1,inplace=True)

# Checking the proportion of good vs bad wine
plt.figure(figsize=(10, 6))
sns.countplot(data["category"], palette="muted")

data

# Separate feature variables and target variable
X = data.drop('category', axis = 1)
Y= data['category']

# Count the number of wines in good category and in bad category 
Y.value_counts()

# Separate cross-validation
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
skf = StratifiedShuffleSplit(n_splits=10,random_state=5)

# TRAINING WITH DEFAULT PARAMETER, GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
nb_classifier = GaussianNB()
all_accuracies = cross_val_score(
    estimator=nb_classifier,
    X=X,
    y=Y,
    cv=skf,
    n_jobs=-1)
print(all_accuracies)
print("mean acc:",np.mean(all_accuracies))

# TUNING PARAMETER using GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
nb_classifier = GaussianNB()
params_NB={'var_smoothing':np.logspace(0,-9, num=100)}
NB = GridSearchCV (estimator = nb_classifier,
                   param_grid = params_NB,
                   cv=skf,
                   verbose = 1,
                   scoring = 'accuracy')
NB.fit(X, Y)
print("best params: ",NB.best_params_)
print("best score: ",NB.best_score_)