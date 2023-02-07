import lazypredict as lz
from lazypredict.Supervised import LazyClassifier
import pickle

# with open('Data/train_test_0.2_0.2s.pickle', 'rb') as handle:
#     data = pickle.load(handle)

# x_train = data['X_train']
# y_train = data['y_train']
# x_test = data['X_test']
# y_test = data['y_test']

# model = LazyClassifier(verbose=0,ignore_warnings=True)
# models, predictions = model.fit(x_train, x_test, y_train, y_test)
# print(models)