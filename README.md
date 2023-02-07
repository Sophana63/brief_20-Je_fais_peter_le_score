# Projet pour le brief sur l'audio

Projet qui contient le code nécessaire au développement du brief audio

## <span style="color: #FF7F50">Sauvegarde des données en pickle<span>

Avant de commencer les tests, je vais créer 2 datasets:

- avec 0.2 secondes en signal
- avec 0.4 secondes en signal

Après avoir calculé toutes les caractéristiques des signaux, je les ai entrainées:

- 20% de jeu de validation
- 10% de jeu de validation

Tout cela va me créer 4 datasets. Pour chacun, je vais les instancier en dictionnaire et ensuite les sauvegarder en pickle.  
Voici les noms des datasets :

- train_test_0.2_0.2s.pickle *# data sur 20% de validation et 0.2 secondes de signaux*
- train_test_0.2_0.4s.pickle *# data sur 20% de validation et 0.4 secondes de signaux*
- train_test_0.1_0.2s.pickle *# data sur 10% de validation et 0.2 secondes de signaux*
- train_test_0.1_0.4s.pickle *# data sur 10% de validation et 0.4 secondes de signaux*


``` python
dico = {"X_train" : learningFeatures_scaled, "X_test" : testFeatures_scaled, "y_train" : learningLabelsStd, "y_test" : testLabelsStd}

with open('Data/train_test_0.2_0.4s.pickle', 'wb') as handle:
    pickle.dump(dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

<span style="color:#FFA500">Le but est de gagner du temps pour chaque test.</span>

Ouvrir le fichier python : [create_pickle.py](create_pickle.py)

Vous pouvez télécharger toutes les données (Cars, Trucks et pickle) grâce à ce lien : [Data.zip](https://drive.google.com/file/d/1xHOTx7ISVFsbX8J4VypmVCZg9xK4VX9V/view?usp=sharing)

<br><hr>

## <span style="color: #FF7F50">Sans changer les paramètres<span>

```
------------------------------------
Score : entre 0.76 et 0.79
------------------------------------
```

<br><hr>


## <span style="color: #FF7F50">En utilisant GridSearchCv<span>

### <span style="color:#9932CC">Code</span>
``` python
from sklearn.model_selection import GridSearchCV

# Paramètre de la grille de recherche de paramètres
param_grid = {
    'C': [1, 5, 10, 50, 100, 200], # Différentes valeurs de C à tester
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], # Différents types de noyaux à tester
    'class_weight': [None, 'balanced'] # Différentes options de poids de classe à tester
    } 

# Initialisation de la grille de recherche
grid = GridSearchCV(model, param_grid, verbose=3, cv=3)

# Entraînement de la grille de recherche sur les données d'apprentissage
grid.fit(learningFeatures_scaled, learningLabelsStd)

# Affichage des meilleurs paramètres trouvés
print("Best parameters found: ", grid.best_params_)
```

Ouvrir le fichier python : [grid_search_cv.py](grid_search_cv.py)



###  <span style="color:#9932CC">Meilleurs paramètres trouvés</span>

Avec 20% de jeu de validation
``` python
# échantillons de 0.2 secondes
Best parameters found:  {'C': 100, 'class_weight': 'balanced', 'kernel': 'rbf'} # test 1, 2 et 3

# échantillons de 0.4 secondes
Best parameters found:  {'C': 50, 'class_weight': None, 'kernel': 'rbf'} # test 1, 2 et 3
```
Avec 10% de jeu de validation
échantillons de 0.4 secondes
``` python
Best parameters found:  {'C': 100, 'class_weight': None, 'kernel': 'rbf'} # test 1
Best parameters found:  {'C': 150, 'class_weight': balanced, 'kernel': 'rbf'} # test 2 et 3
```

###  <span style="color:#9932CC">Resultats:</span>

Avec 20% de jeu de validation
``` python
# échantillons de 0.2 secondes
[CV 3/3] END C=100, class_weight=balanced, kernel=rbf;, score=0.867 # test 1
[CV 3/3] END C=100, class_weight=balanced, kernel=rbf;, score=0.866 # test 2
[CV 3/3] END C=100, class_weight=balanced, kernel=rbf;, score=0.867 # test 3

# échantillons de 0.4 secondes
[CV 3/3] END C=50, class_weight=balanced, kernel=rbf;, score=0.894 # test 1, 2, 3
```

Avec 10% de jeu de validation
``` python
# échantillons de 0.2 secondes
[CV 3/3] END C=100, class_weight=balanced, kernel=rbf;, score=0.879  # test 1, 2, 3

# échantillons de 0.4 secondes
[CV 3/3] END C=100, class_weight=None, kernel=rbf;, score=0.889 # test 1
[CV 1/3] END C=150, class_weight=balanced, kernel=rbf;, score=0.896 # test 2 et 3

```

## <span style="color: #FF7F50">En utilisant MLPClassifier<span>

### <span style="color:#9932CC">Code du modèle</span>
``` python
model = MLPClassifier(hidden_layer_sizes=(500,500), max_iter=500, activation="tanh", solver="adam")
```

### <span style="color:#9932CC">Resultats</span>
``` python
# 10% validation & 0.2 secondes
Score sur les données entraînement: 1.000
Score sur les données de test: 0.940 # test 1
Score sur les données de test: 0.904 # test 2
Score sur les données de test: 0.920 # test 3

# 10% validation & 0.4 secondes
Score sur les données de test: 0.939 # test 1
Score sur les données de test: 0.929 # test 2
Score sur les données de test: 0.939 # test 3


# 20% validation & 0.2 secondes
Score sur les données de test: 0.889 # test 1
Score sur les données de test: 0.884 # test 2
Score sur les données de test: 0.912 # test 3

# 20% validation & 0.2 secondes
Score sur les données de test: 0.874 # test 1
Score sur les données de test: 0.869 # test 2
Score sur les données de test: 0.874 # test 3

```

    Le meilleur score trouvé est sur le jeu de validation le plus petit, 10%. La durée des signaux n'a pas vraiment d'impact.

Ouvrir le fichier python : [mlp.py](mlp.py)

<br><hr>

## <span style="color: #FF7F50">Deep Learning avec le réseau CNN (convolution)<span>

J'ai construit un modèle avec 3 couches de convolution et 2 couches denses. 
Au moment de compiler, j'ai utilisé la metric **accuracy** et l'optimizer **adam** .

Voici le résultat:
``` python
Test accuracy: 0.5326633453369141
```

Ouvrir le fichier python : [CNN.py](cnn.py)

<br><hr>

## <span style="color: #FF7F50">Random Forest<span>

Voici les hyperparamètres que j'ai utilisé pour optimiser la précision:

``` python
model = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=20)
```

Pour 20% de jeu de validation
``` python
Accuracy avec 0.2 secondes: 0.8841
Accuracy avec 0.4 secondes: 0.8889
```

Pour 10% de jeu de validation
``` python
Accuracy avec 0.2 secondes: 0.8945
Accuracy avec 0.4 secondes: 0.9192 
```

    La meilleur précision s'obtient avec un jeu de validation de 10% pour le temps de signal de 0.4 secondes 


Ouvrir le fichier python : [random_forest.py](random_forest.py)

<br><hr>

## <span style="color: #FF7F50">CatBoost<span>

``` python
# 10% test & 0.2s
bestTest = 0.207076661
bestIteration = 274
---------------------
CatBoost Accuracy: 0.92462
---------------------
RMSE: 0.27
R2: 0.70


# 10% test & 0.4s
bestTest = 0.2225615188
bestIteration = 216
Shrink model to first 217 iterations.
---------------------
CatBoost Accuracy: 0.92929
---------------------
RMSE: 0.27
R2: 0.72

# 20% test & 0.2s
bestTest = 0.2488954402
bestIteration = 274
---------------------
CatBoost Accuracy: 0.89672
---------------------
RMSE: 0.32
R2: 0.59

# 20% test & 0.4s
bestTest = 0.2460288818
bestIteration = 274
---------------------
CatBoost Accuracy: 0.91414
---------------------
RMSE: 0.29
R2: 0.66
```

Ouvrir le fichier python : [catboost_clf.py](catboost_clf.py)

<br><hr>

## <span style="color: #FF7F50">CONCLUSION<span>

Voici les meilleurs prédictions pour chaque dataset :

### <span style="color:#9932CC">20% de validation 0.2s de signal</span>
``` python
MLPClassifier >>>> 0.896
```

### <span style="color:#9932CC">20% de validation 0.4s de signal</span>
``` python
MLPClassifier >>>> 0.914
```

### <span style="color:#9932CC">10% de validation 0.2s de signal</span>
``` python
Catboost  >>>> 0.940
```

### <span style="color:#9932CC">10% de validation 0.4s de signal</span>
``` python
Catboost  >>>> 0.939
```