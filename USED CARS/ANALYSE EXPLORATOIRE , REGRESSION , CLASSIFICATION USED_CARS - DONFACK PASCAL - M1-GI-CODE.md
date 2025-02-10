# ANALYSE EXPLORATOIRE ET CLUSTURING  SUR LE  DATASET 'USED CARS' : DONFACK PASCAL : M1-GI

## Importation des librairies et chargement du dataset


```python
# cd "C:\Users\donfa\OneDrive\Desktop\DEVOIR ML"
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('used_cars.csv')

df.head()
```


```python
print(f"Le dataset contient {df.shape[0]} lignes et {df.shape[1]} colonnes.")
```

### Informations sur le dataset


```python
df.info()
```

### graphique de distribution des colones du dataset , dans le but de ressortir certaines tendances


```python
# This code shows distribution plot for all other columns
for column in df:
    sns.displot(x=column, data=df)
```


```python
# sns.pairplot(df,hue='model_year')
```

## Nettoyage des données

### On remarque que dans ce dataset le prix n'est pas sous la bonne forme


```python
# Suppression des espaces inutiles dans les noms de colonnes
df.columns = df.columns.str.strip()

# Conversion de la colonne 'price' en numérique
df['price'] = df['price'].replace('[$,]', '', regex=True).astype(float)

# Vérification des valeurs manquantes
missing_values = df.isnull().sum()

missing_values
```

### On constate que la colone milage n'est pas bonne non plus


```python
# Transformation de la colonne mileage
df['milage'] = df['milage'].str.replace(',', '').str.replace(' mi.', '').astype(float)

df['milage'].head()
```


```python
sns.pairplot(df,hue='model_year')
```


```python
df.info()
```


```python
# Suppression des valeurs manquantes
df = df.dropna()
```

## Analyse exploratoire


```python
# Description statistique des variables numériques
df.describe()
```




```python
# Description des variables catégorielles
df.describe(include=['object'])
```

#### La marque la plus fréquente est Ford


```python
df['brand'].value_counts()
```

#### Il y a 52 marques différentes dans le dataset.

### Visualisation des données


```python
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution des prix des voitures')
plt.xlabel('Prix')
plt.ylabel('Fréquence')
plt.show()
```

#### La distribution des prix des voitures est asymétrique à droite. les voitures les moins chers sont les plus nombreuses

### quelques stats interessantes



```python
most_expensive_car = df.loc[df['price'].idxmax()]
print("Voiture la plus chère :")
print(most_expensive_car[['brand', 'model', 'price']])
```


```python
# Voiture la moins chère
cheapest_car = df.loc[df['price'].idxmin()]
print("\nVoiture la moins chère :")
print(cheapest_car[['brand', 'model', 'price']])
```


```python
brand_counts = df['brand'].value_counts()
print("\nNombre de voitures par marque :")
print(brand_counts)
```


```python
average_price_by_brand = df.groupby('brand')['price'].mean().sort_values(ascending=False)
print("\nPrix moyen par marque :")
print(average_price_by_brand)
```


```python
accident_counts = df['accident'].value_counts()
print("\nNombre de voitures accidentées vs non accidentées :")
print(accident_counts)
```


```python
average_price_by_accident = df.groupby('accident')['price'].mean()
print("\nPrix moyen des voitures accidentées vs non accidentées :")
print(average_price_by_accident)
```



### Corrélation entre les variables numériques


```python
# Corrélation entre les variables numériques
correlation_matrix = df[['price','model_year']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation entre les variables numériques')
plt.show()
```

Il y a qu'une tres faible correlation positive entre le prix et l'année de fabrication de la voiture


```python
# Analyse des variables catégorielles
categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
df_saved=df[['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title','price','model_year','milage']]
df=df[['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title','price','model_year']]

for col in categorical_columns:
    print(f"\nValeurs uniques pour {col}: {df[col].unique()}")
```

## CLUSTURING


```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Encodage des variables catégorielles avec get_dummies
df_encoded = pd.get_dummies(df, columns=['brand', 'accident'], drop_first=True)

# Sélection des caractéristiques pour le clustering
X_cluster = df_encoded[['price'] + list(df_encoded.filter(like='brand_')) + list(df_encoded.filter(like='model_'))]

# Standardisation des données
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Méthode du coude pour déterminer le nombre optimal de clusters
inertia = []
for k in range(1, 11):  # Test de 1 à 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)  # Inertie pour chaque nombre de clusters

# Visualisation de la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

optimal_k = 8  

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='brand', hue='cluster', data=df, palette='viridis')
plt.title('Clustering des voitures')
plt.xlabel('Prix')
plt.ylabel('Marque')
plt.show()
```

## PRETRAITEMENT DES DONNEES 


```python
# Encodage des variables catégorielles avec get_dummies
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Conversion des booléens (True/False) en entiers (1/0)
df = df.astype(int)

df.head()
```

## REGRESSION LINEAIRE


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Sélection des caractéristiques et de la cible
X = df.drop('price', axis=1)
y = df['price']

# Standardisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calcul de l'erreur en pourcentage (MAPE - Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

# Affichage des résultats
print(f'MSE: {mse:.2f}')
print(f'R2: {r2:.2f}')
print(f'MAPE: {mape:.2f}%')
```

### La regression linaire semble ne pas etre une bonne methode de prediction du prix pour ce dataset

### testons avec les deux colones numeriques milage et price
#### Regression linaire linaire 


```python
from sklearn.metrics import mean_absolute_error
df=df_saved
X = df[['milage']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

y_pred_lin = lin_reg.predict(X_test_scaled)

# Évaluation
def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    rel_error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Erreur relative %

    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² : {r2:.2f}")
    print(f"Erreur relative moyenne : {rel_error:.2f} %")

print("=== Régression Linéaire ===")
eval_metrics(y_test, y_pred_lin)

```

### Regression polynomiale


```python

```


```python
from sklearn.preprocessing import PolynomialFeatures

degrees = [2, 3, 4]  # Tester plusieurs degrés

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)

    y_pred_poly = poly_reg.predict(X_test_poly)

    print(f"\n=== Régression Polynomiale (degré {d}) ===")
    eval_metrics(y_test, y_pred_poly)

# ===== 4. Visualisation =====
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X_test, y_pred_lin, color='red', label='Régression Linéaire')

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(scaler.transform(X))
    y_poly_pred = LinearRegression().fit(X_poly, y).predict(X_poly)
    plt.plot(X, y_poly_pred, label=f'Poly deg {d}')

plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend()
plt.show()
```

### Nous observons cependant que la force de la distribution est proche de la fonction exponentielle 

#### tentons une regression avec ce model


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Variables X (mileage) et y (price)
X = df[['milage']]
y = df['price']

# Transformation logarithmique de y
y_log = np.log(y)

# Division en train/test
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 2. Régression Linéaire sur log(y) =====
exp_reg = LinearRegression()
exp_reg.fit(X_train_scaled, y_train_log)

# Prédictions (dans l’espace log)
y_pred_log = exp_reg.predict(X_test_scaled)

# Transformation inverse (exponentielle) pour revenir à l’échelle normale
y_pred_exp = np.exp(y_pred_log)

# Évaluation
def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    rel_error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Erreur relative %

    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² : {r2:.2f}")
    print(f"Erreur relative moyenne : {rel_error:.2f} %")

print("=== Régression Exponentielle ===")
eval_metrics(np.exp(y_test_log), y_pred_exp)

# ===== 3. Visualisation =====
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X_test, y_pred_exp, color='green', label='Régression Exponentielle')

plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend()
plt.show()

```

### La regression exponentielle semble etre la meilleure methode pour predire le prix en fonction du kilometrage


```python

```
