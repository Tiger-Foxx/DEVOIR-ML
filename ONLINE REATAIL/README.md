# ANALYSE EXPLORATOIRE , CLASSIFICATION ONLINE RETAIL - DONFACK PASCAL - M1-GI


```python
# cd "C:\Users\donfa\OneDrive\Desktop\DEVOIR ML\ONLINE REATAIL"
```

    C:\Users\donfa\OneDrive\Desktop\DEVOIR ML\ONLINE REATAIL
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Lecture du dataset
df = pd.read_excel('Online Retail.xlsx')
```

# PARTIE 1: EXPLORATION ET NETTOYAGE DES DONNÉES


```python
# Affichage des informations de base du dataset
print("=== INFORMATIONS SUR LE DATASET ===")
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")
print("\nTypes des colonnes:")
df.info()
```

    === INFORMATIONS SUR LE DATASET ===
    Nombre de lignes : 541909
    Nombre de colonnes : 8
    
    Types des colonnes:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541909 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    541909 non-null  object        
     1   StockCode    541909 non-null  object        
     2   Description  540455 non-null  object        
     3   Quantity     541909 non-null  int64         
     4   InvoiceDate  541909 non-null  datetime64[ns]
     5   UnitPrice    541909 non-null  float64       
     6   CustomerID   406829 non-null  float64       
     7   Country      541909 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 33.1+ MB
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("\nStatistiques descriptives :")
df.describe()
```

    
    Statistiques descriptives :
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>541909.000000</td>
      <td>541909</td>
      <td>541909.000000</td>
      <td>406829.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.552250</td>
      <td>2011-07-04 13:34:57.156386048</td>
      <td>4.611114</td>
      <td>15287.690570</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-80995.000000</td>
      <td>2010-12-01 08:26:00</td>
      <td>-11062.060000</td>
      <td>12346.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>2011-03-28 11:34:00</td>
      <td>1.250000</td>
      <td>13953.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2011-07-19 17:17:00</td>
      <td>2.080000</td>
      <td>15152.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>2011-10-19 11:27:00</td>
      <td>4.130000</td>
      <td>16791.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>2011-12-09 12:50:00</td>
      <td>38970.000000</td>
      <td>18287.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>218.081158</td>
      <td>NaN</td>
      <td>96.759853</td>
      <td>1713.600303</td>
    </tr>
  </tbody>
</table>
</div>



### Nettoyage des données


```python
# =============================================================================
# Gestion des valeurs manquantes, incohérences et doublons
#=============================================================================
```


```python
print("\nValeurs manquantes par colonne :")
df.isnull().sum()
```

    
    Valeurs manquantes par colonne :
    




    InvoiceNo           0
    StockCode           0
    Description      1454
    Quantity            0
    InvoiceDate         0
    UnitPrice           0
    CustomerID     135080
    Country             0
    dtype: int64




```python
# Suppression des lignes sans CustomerID
df_clean = df.dropna()
```


```python
# Suppression des lignes avec des quantités négatives ou prix unitaire nul
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
```


```python
# forme du dataset apres nettoyage
print(f"Nombre de lignes après nettoyage : {df_clean.shape[0]}")
print(f"Nombre de colonnes après nettoyage : {df_clean.shape[1]}")
df_clean.info()
```

    Nombre de lignes après nettoyage : 397884
    Nombre de colonnes après nettoyage : 8
    <class 'pandas.core.frame.DataFrame'>
    Index: 397884 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    397884 non-null  object        
     1   StockCode    397884 non-null  object        
     2   Description  397884 non-null  object        
     3   Quantity     397884 non-null  int64         
     4   InvoiceDate  397884 non-null  datetime64[ns]
     5   UnitPrice    397884 non-null  float64       
     6   CustomerID   397884 non-null  object        
     7   Country      397884 non-null  object        
    dtypes: datetime64[ns](1), float64(1), int64(1), object(5)
    memory usage: 27.3+ MB
    

## Feature Engineering


```python
# Calcul du montant total par transaction
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
```


```python
print("\n=== STATISTIQUES DESCRIPTIVES ===")
(df_clean.describe())
```

    
    === STATISTIQUES DESCRIPTIVES ===
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>TotalAmount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>397884.000000</td>
      <td>397884</td>
      <td>397884.000000</td>
      <td>397884.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.988238</td>
      <td>2011-07-10 23:41:23.511023360</td>
      <td>3.116488</td>
      <td>22.397000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2010-12-01 08:26:00</td>
      <td>0.001000</td>
      <td>0.001000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2011-04-07 11:12:00</td>
      <td>1.250000</td>
      <td>4.680000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>2011-07-31 14:39:00</td>
      <td>1.950000</td>
      <td>11.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.000000</td>
      <td>2011-10-20 14:33:00</td>
      <td>3.750000</td>
      <td>19.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>2011-12-09 12:50:00</td>
      <td>8142.750000</td>
      <td>168469.600000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>179.331775</td>
      <td>NaN</td>
      <td>22.097877</td>
      <td>309.071041</td>
    </tr>
  </tbody>
</table>
</div>



## Analyse des ventes par pays


```python
country_sales = df_clean.groupby('Country')['TotalAmount'].agg(['sum', 'count']).sort_values('sum', ascending=False)
print("\n=== TOP 10 PAYS PAR VENTES ===")
country_sales.head(10)

```

    
    === TOP 10 PAYS PAR VENTES ===
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United Kingdom</th>
      <td>7308391.554</td>
      <td>354321</td>
    </tr>
    <tr>
      <th>Netherlands</th>
      <td>285446.340</td>
      <td>2359</td>
    </tr>
    <tr>
      <th>EIRE</th>
      <td>265545.900</td>
      <td>7236</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>228867.140</td>
      <td>9040</td>
    </tr>
    <tr>
      <th>France</th>
      <td>209024.050</td>
      <td>8341</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>138521.310</td>
      <td>1182</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>61577.110</td>
      <td>2484</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>56443.950</td>
      <td>1841</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>41196.340</td>
      <td>2031</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>38378.330</td>
      <td>451</td>
    </tr>
  </tbody>
</table>
</div>



# Analyse RFM (Recency, Frequency, Monetary)



```python
# Conversion de InvoiceDate en datetime si nécessaire
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# Calcul de la date la plus récente
max_date = df_clean['InvoiceDate'].max()

```


```python
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalAmount': 'sum'   # Monetary
})
```


```python
rfm.columns = ['Recency', 'Frequency', 'Monetary']
```

# PARTIE 3: VISUALISATIONS


```python
# Distribution des montants des transactions
plt.figure(figsize=(10, 6))
plt.hist(df_clean['TotalAmount'], bins=50)
plt.title('Distribution des montants des transactions')
plt.xlabel('Montant')
plt.ylabel('Fréquence')
plt.show()
```


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_23_0.png)
    



```python
df_clean['TotalAmount'].describe()
```




    count    397884.000000
    mean         22.397000
    std         309.071041
    min           0.001000
    25%           4.680000
    50%          11.800000
    75%          19.800000
    max      168469.600000
    Name: TotalAmount, dtype: float64




```python
# Distribution des quantités commandées
plt.figure(figsize=(10, 6))
plt.hist(df_clean['Quantity'],)
plt.title('Distribution des quantités commandées')
plt.xlabel('Quantité')
plt.ylabel('Fréquence')
plt.show()
```


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_25_0.png)
    



```python
df_clean['Quantity'].describe()
```




    count    397884.000000
    mean         12.988238
    std         179.331775
    min           1.000000
    25%           2.000000
    50%           6.000000
    75%          12.000000
    max       80995.000000
    Name: Quantity, dtype: float64




```python
# Préparation des données pour le clustering
# Standardisation des variables RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Détermination du nombre optimal de clusters avec la méthode du coude
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertias.append(kmeans.inertia_)
```


```python
# Visualisation de la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour la détermination du nombre optimal de clusters')
plt.show()
```


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_28_0.png)
    



```python
# Application du clustering avec le nombre optimal de clusters (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyse des clusters
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)
print("\n=== ANALYSE DES CLUSTERS ===")
print(cluster_analysis)
```

    
    === ANALYSE DES CLUSTERS ===
             Recency  Frequency   Monetary
    Cluster                               
    0          40.38     103.09    2028.83
    1         246.31      27.79     637.32
    2           3.69    2565.31  126118.31
    

## Visualisation des clusters


```python
# Visualisation 3D des clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(rfm['Recency'], 
                    rfm['Frequency'], 
                    rfm['Monetary'],
                    c=rfm['Cluster'],
                    cmap='viridis')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.colorbar(scatter)
plt.title('Visualisation 3D des clusters de clients')
plt.show()
```


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20CLASSIFICATION%20ONLINE%20RETAIL%20-%20DONFACK%20PASCAL%20-%20M1-GI_31_0.png)
    


## Interpretation 3D des clusters de clients


```python
print("\n=== INTERPRÉTATION DES CLUSTERS ===")
for cluster in range(3):
    cluster_size = (rfm['Cluster'] == cluster).sum()
    cluster_pct = cluster_size / len(rfm) * 100
    print(f"\nCluster {cluster}:")
    print(f"Nombre de clients: {cluster_size} ({cluster_pct:.1f}%)")
    print(f"Récence moyenne: {cluster_analysis.loc[cluster, 'Recency']:.1f} jours")
    print(f"Fréquence moyenne: {cluster_analysis.loc[cluster, 'Frequency']:.1f} commandes")
    print(f"Montant moyen: {cluster_analysis.loc[cluster, 'Monetary']:.2f} £")
```

    
    === INTERPRÉTATION DES CLUSTERS ===
    
    Cluster 0:
    Nombre de clients: 3245 (74.8%)
    Récence moyenne: 40.4 jours
    Fréquence moyenne: 103.1 commandes
    Montant moyen: 2028.83 £
    
    Cluster 1:
    Nombre de clients: 1080 (24.9%)
    Récence moyenne: 246.3 jours
    Fréquence moyenne: 27.8 commandes
    Montant moyen: 637.32 £
    
    Cluster 2:
    Nombre de clients: 13 (0.3%)
    Récence moyenne: 3.7 jours
    Fréquence moyenne: 2565.3 commandes
    Montant moyen: 126118.31 £
    
