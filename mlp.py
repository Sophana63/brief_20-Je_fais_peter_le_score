import pickle
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

with open('Data/train_test_0.1_0.4s.pickle', 'rb') as handle:
    data = pickle.load(handle)

x_train = data['X_train']
y_train = data['y_train']
x_test = data['X_test']
y_test = data['y_test']


model = MLPClassifier(hidden_layer_sizes=(500,500), max_iter=650, activation="tanh", solver="adam")
model.fit(x_train, y_train)

print("Score sur les données d'entraînement: {:.3f}".format(model.score(x_train, y_train)))
print("Score sur les données de test: {:.3f}".format(model.score(x_test, y_test)))


# -----------------------------------------------------------------------
# test sur tous les activations
# -----------------------------------------------------------------------
# max_iters = [100, 200, 300, 400, 500, 600]
# activations = ['identity', 'logistic', 'tanh', 'relu']

# for activation in activations:
#     test_score = []
#     for max_iter in max_iters:
#         model = MLPClassifier(hidden_layer_sizes=(500,500), max_iter=max_iter, activation=activation, solver="adam")
#         model.fit(x_train, y_train)
#         test_score.append(model.score(x_test, y_test))
#     plt.plot(max_iters, test_score, label=activation)

# plt.xlabel('Max Iterations')
# plt.ylabel('Test Score')
# plt.legend()
# plt.show()


# -----------------------------------------------------------------------
# test sur tous les solvers
# -----------------------------------------------------------------------
# max_iters = [400, 500, 600, 700, 800]
# solvers = ['adam', 'sgd', 'lbfgs']

# for solver in solvers:
#     test_score = []
#     for max_iter in max_iters:
#         model = MLPClassifier(hidden_layer_sizes=(500,500), max_iter=max_iter, activation='tanh', solver=solver)
#         model.fit(x_train, y_train)
#         test_score.append(model.score(x_test, y_test))
#     plt.plot(max_iters, test_score, label=solver)

# plt.xlabel('Max Iterations')
# plt.ylabel('Test Score')
# plt.legend()
# plt.show()

