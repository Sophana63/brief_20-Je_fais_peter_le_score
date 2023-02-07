import pickle
import catboost as cb
from catboost import  CatBoostClassifier
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

with open('Data/train_test_0.2_0.4s.pickle', 'rb') as handle:
    data = pickle.load(handle)

x_train = data['X_train']
y_train = data['y_train']
x_test = data['X_test']
y_test = data['y_test']

clf = CatBoostClassifier(
    iterations=275,
    depth =10,
    custom_loss=['AUC', 'Accuracy']
)

clf.fit(
    x_train, y_train,
    eval_set=(x_test, y_test),
    verbose=False,
    plot=True
)

pred = clf.predict(x_test)
r2 = r2_score(y_test, pred)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))

print('---------------------')
print('CatBoost Accuracy:', clf.score(x_test, y_test))
print('---------------------')
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.2f}'.format(r2))
