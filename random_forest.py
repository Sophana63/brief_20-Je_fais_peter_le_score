import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open('Data/train_test_0.1_0.4s.pickle', 'rb') as handle:
    data = pickle.load(handle)

x_train = data['X_train']
y_train = data['y_train']
x_test = data['X_test']
y_test = data['y_test']

model = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=20)

model.fit(x_train, y_train)
preds = model.predict(x_test)

print('Accuracy:  {:.4f}'.format(accuracy_score(y_test, preds)))