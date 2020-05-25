#Partie 1 - Préparation des données
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
#panda pour la lecture de csv
import pandas as pd
import ga
from model_generator import model_generator

dataset = pd.read_csv('./data/car_data.csv')
dataset.dropna(axis=0, how='any')

# On supprime la troisième colonne, que l'on souhaite prédire et la première
X = np.array(dataset.iloc[:, 1:].values)
X = np.delete(X, 1, 1)

y = dataset.iloc[:, 2].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#maintenant je renseigne le fait que ma colonne 1 est une colonne de type variable non catégorique, je dois donc la travailler
#je transforme mes pays en valeur numerique
labelencoder_X_1 = LabelEncoder()
X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])

#ma colonne deux qui contient le genre va aussi avoir un traitement
#je transforme mes genre en valeur numerique
labelencoder_X_2 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])


labelencoder_X_3 = LabelEncoder()
X[:, 5] = labelencoder_X_3.fit_transform(X[:, 5])

transformer1 = ColumnTransformer(
    transformers=[
        ("Fuel, seller and transmission",        # Just a name
          OneHotEncoder(), # The transformer class
          [3,4,5]            # The column(s) to be applied on.
          )
    ], remainder='passthrough'
)

X = transformer1.fit_transform(X)

# Une colonne en trop est ajoutée à chaque transformation, on les supprime donc (3ème colonne de fuel et 2ème colonne pour seller et transmission)
X = np.delete(X, [2, 4, 6], 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%

population = ga.get_first_pop(10, False)

for gen in range(1):
    
    scored_pop = []
    for pop in range(len(population)):
        try: 
            model = model_generator(population[pop])
            
            history = model.fit(X_train, y_train, batch_size=population[0]['batch_size'], epochs=population[0]['epochs'])
            
            y_pred = model.predict(X_test)
            
            score = model.evaluate(X_test, y_test, batch_size=1)
            scored_pop.append([population[pop], score])
            # for gen in range(100):
        except: 
            scored_pop.append([population[pop], 9999])
            
            
#%%
            
for a in scored_pop:
    print(a[1])