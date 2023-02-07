from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt

# ouvre le fichier pickle
# changez le nom pour tester avec plusieus données différentes
with open('Data/train_test_0.2_0.4s.pickle', 'rb') as handle:
    data = pickle.load(handle)

model = svm.SVC()

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
param_grid = {'C': [ 25, 50, 75, 100, 125, 150], 'kernel': kernels, 'class_weight': [None, 'balanced']}
grid = GridSearchCV(model, param_grid, verbose=3, cv=3)
grid.fit(data['X_train'], data['y_train'])
print("Best parameters found: ", grid.best_params_)

for kernel in kernels:
    indices = [i for i, x in enumerate(grid.cv_results_['params']) if x['kernel'] == kernel]
    means = [grid.cv_results_['mean_test_score'][i] for i in indices]
    C_values = [grid.cv_results_['params'][i]['C'] for i in indices]
    plt.plot(C_values, means, label=kernel)

plt.xlabel('C')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.show()

