import pickle
from sklearn.neural_network import MLPClassifier

with open('Data/train_test_0.1_0.4s.pickle', 'rb') as handle:
    data = pickle.load(handle)

x_train = data['X_train']
y_train = data['y_train']
x_test = data['X_test']
y_test = data['y_test']


#X_train, X_test, y_train, y_test = train_test_split(data['features'], data['learning_labels'], test_size=0.2, random_state=2)

model = MLPClassifier(hidden_layer_sizes=(500,500), max_iter=500, activation="tanh", solver="adam")
model.fit(x_train, y_train)

print("Score sur les données d'entraînement: {:.3f}".format(model.score(x_train, y_train)))
print("Score sur les données de test: {:.3f}".format(model.score(x_test, y_test)))

