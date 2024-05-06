import os
import pandas as pd
from DataHandling import create_dataframe
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
import pickle

debug = 0

website_to_csv_names = {}
website_key_pair = {}
website_names = []
i = 0

more_sites = True
while(more_sites):
    website = input("What website are these CSV files from: ")
    folder = input("What folder will we find your CSVs this website in: ")
    website_names.append(website)
    website_key_pair[website] = i
    i += 1
    if website not in website_to_csv_names.keys():
        website_to_csv_names[website] = []
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if debug > 0:
                print("Found: " + f)
            website_to_csv_names[website].append(f)
    go = input("Do you have more data [y/n]: ")
    more_sites = (go=='y')
    
print('\nData collection finished\n')

dataframe = pd.DataFrame()
for name in website_key_pair.keys():
    print('Building dataframe for: ' + name)
    new_dataframe = create_dataframe(website_to_csv_names[name], website_key_pair[name])
    if dataframe.empty:
        dataframe = new_dataframe
    else:
        dataframe = pd.concat([dataframe, new_dataframe])

print("Built dataframes\n")

display(dataframe)

print("\nSplitting data")

dataframe.to_csv('total_data.csv')

X_train, X_test, y_train, y_test = train_test_split(dataframe, dataframe['Website'])

print('Building fitting and testing regression models')
linear = linear_model.SGDClassifier()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)
print(y_test)
print(y_pred)
print("\nLinear Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nLinear Regression Precision:", precision_score(y_test, y_pred, average=None))
print("\nLinear Regression F1:", f1_score(y_test, y_pred, average=None))

logistic = linear_model.LogisticRegressionCV(max_iter=600)
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print("\nLogisticRegression Accuracy:", accuracy_score(y_test, y_pred))
print("\nLogisticRegression Precision:", precision_score(y_test, y_pred, average=None))
print("\nLogisticRegression F1:", f1_score(y_test, y_pred, average=None))

print("\nBuilding training and testing trees")
#build trees
decision_tree = DecisionTreeClassifier(random_state = 0)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
pickle.dumps(decision_tree)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nDecision Tree Precision:", precision_score(y_test, y_pred, average=None))
print("\nDecision Tree F1:", f1_score(y_test, y_pred, average=None))

forest = RandomForestClassifier(random_state = 0)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
pickle.dumps(forest)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nRandom Forest Precision:", precision_score(y_test, y_pred, average=None))
print("\nRandom Forest F1:", f1_score(y_test, y_pred, average=None))

boosting = GradientBoostingClassifier(random_state=0)
boosting.fit(X_train, y_train)
y_pred = forest.predict(X_test)
pickle.dumps(boosting)
print("\nBoosting Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nBoosting Tree Precision:", precision_score(y_test, y_pred, average=None))
print("\nBoosting Tree F1:", f1_score(y_test, y_pred, average=None))

from sklearn.neighbors import KNeighborsClassifier

knearest = KNeighborsClassifier()
knearest.fit(X_train, y_train)

y_pred_f = knearest.predict(X_test)

print("\nK Neighbors Accuracy:", accuracy_score(y_test, y_pred_f))
print("\nK Neighbors Precision:", precision_score(y_test, y_pred_f, average=None))
print("\nK Neighbors F1:", f1_score(y_test, y_pred_f, average=None))


prediction = pd.DataFrame(forest.predict(X_test))
answers = pd.DataFrame(y_test)

prediction.to_csv('predictions.csv')
answers.to_csv('answers.csv')