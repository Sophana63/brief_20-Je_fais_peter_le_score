from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle

# ouvre le fichier pickle
# changez le nom pour tester avec plusieus données différentes
with open('Data/train_test_0.1_0.4s.pickle', 'rb') as handle:
    data = pickle.load(handle)

model = svm.SVC()

param_grid = {'C': [10, 30, 50, 100, 130, 150], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'class_weight': [None, 'balanced']}
grid = GridSearchCV(model, param_grid, verbose=3, cv=3)
grid.fit(data['X_train'], data['y_train'])
print("Best parameters found: ", grid.best_params_)