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
      <th>brand</th>
      <th>model</th>
      <th>model_year</th>
      <th>milage</th>
      <th>fuel_type</th>
      <th>engine</th>
      <th>transmission</th>
      <th>ext_col</th>
      <th>int_col</th>
      <th>accident</th>
      <th>clean_title</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ford</td>
      <td>Utility Police Interceptor Base</td>
      <td>2013</td>
      <td>51,000 mi.</td>
      <td>E85 Flex Fuel</td>
      <td>300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capa...</td>
      <td>6-Speed A/T</td>
      <td>Black</td>
      <td>Black</td>
      <td>At least 1 accident or damage reported</td>
      <td>Yes</td>
      <td>$10,300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hyundai</td>
      <td>Palisade SEL</td>
      <td>2021</td>
      <td>34,742 mi.</td>
      <td>Gasoline</td>
      <td>3.8L V6 24V GDI DOHC</td>
      <td>8-Speed Automatic</td>
      <td>Moonlight Cloud</td>
      <td>Gray</td>
      <td>At least 1 accident or damage reported</td>
      <td>Yes</td>
      <td>$38,005</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lexus</td>
      <td>RX 350 RX 350</td>
      <td>2022</td>
      <td>22,372 mi.</td>
      <td>Gasoline</td>
      <td>3.5 Liter DOHC</td>
      <td>Automatic</td>
      <td>Blue</td>
      <td>Black</td>
      <td>None reported</td>
      <td>NaN</td>
      <td>$54,598</td>
    </tr>
    <tr>
      <th>3</th>
      <td>INFINITI</td>
      <td>Q50 Hybrid Sport</td>
      <td>2015</td>
      <td>88,900 mi.</td>
      <td>Hybrid</td>
      <td>354.0HP 3.5L V6 Cylinder Engine Gas/Electric H...</td>
      <td>7-Speed A/T</td>
      <td>Black</td>
      <td>Black</td>
      <td>None reported</td>
      <td>Yes</td>
      <td>$15,500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Audi</td>
      <td>Q3 45 S line Premium Plus</td>
      <td>2021</td>
      <td>9,835 mi.</td>
      <td>Gasoline</td>
      <td>2.0L I4 16V GDI DOHC Turbo</td>
      <td>8-Speed Automatic</td>
      <td>Glacier White Metallic</td>
      <td>Black</td>
      <td>None reported</td>
      <td>NaN</td>
      <td>$34,999</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"Le dataset contient {df.shape[0]} lignes et {df.shape[1]} colonnes.")
```

    Le dataset contient 4009 lignes et 12 colonnes.
    

### Informations sur le dataset


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4009 entries, 0 to 4008
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   brand         4009 non-null   object
     1   model         4009 non-null   object
     2   model_year    4009 non-null   int64 
     3   milage        4009 non-null   object
     4   fuel_type     3839 non-null   object
     5   engine        4009 non-null   object
     6   transmission  4009 non-null   object
     7   ext_col       4009 non-null   object
     8   int_col       4009 non-null   object
     9   accident      3896 non-null   object
     10  clean_title   3413 non-null   object
     11  price         4009 non-null   object
    dtypes: int64(1), object(11)
    memory usage: 376.0+ KB
    

### graphique de distribution des colones du dataset , dans le but de ressortir certaines tendances


```python
# This code shows distribution plot for all other columns
for column in df:
    sns.displot(x=column, data=df)
```


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_0.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_1.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_2.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_3.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_4.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_5.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_6.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_7.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_8.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_9.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_10.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_8_11.png)
    



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




    brand             0
    model             0
    model_year        0
    milage            0
    fuel_type       170
    engine            0
    transmission      0
    ext_col           0
    int_col           0
    accident        113
    clean_title     596
    price             0
    dtype: int64



### On constate que la colone milage n'est pas bonne non plus


```python
# Transformation de la colonne mileage
df['milage'] = df['milage'].str.replace(',', '').str.replace(' mi.', '').astype(float)

df['milage'].head()
```




    0    51000.0
    1    34742.0
    2    22372.0
    3    88900.0
    4     9835.0
    Name: milage, dtype: float64




```python
sns.pairplot(df,hue='model_year')
```




    <seaborn.axisgrid.PairGrid at 0x28f8244e450>




    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_14_1.png)
    



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4009 entries, 0 to 4008
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   brand         4009 non-null   object 
     1   model         4009 non-null   object 
     2   model_year    4009 non-null   int64  
     3   milage        4009 non-null   float64
     4   fuel_type     3839 non-null   object 
     5   engine        4009 non-null   object 
     6   transmission  4009 non-null   object 
     7   ext_col       4009 non-null   object 
     8   int_col       4009 non-null   object 
     9   accident      3896 non-null   object 
     10  clean_title   3413 non-null   object 
     11  price         4009 non-null   float64
    dtypes: float64(2), int64(1), object(9)
    memory usage: 376.0+ KB
    


```python
# Suppression des valeurs manquantes
df = df.dropna()
```

## Analyse exploratoire


```python
# Description statistique des variables numériques
df.describe()
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
      <th>model_year</th>
      <th>milage</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3269.000000</td>
      <td>3269.000000</td>
      <td>3.269000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.601407</td>
      <td>72126.951973</td>
      <td>4.124113e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.152181</td>
      <td>53387.413623</td>
      <td>8.304604e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1974.000000</td>
      <td>100.000000</td>
      <td>2.000000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2011.000000</td>
      <td>30450.000000</td>
      <td>1.550000e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2016.000000</td>
      <td>62930.000000</td>
      <td>2.800000e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2019.000000</td>
      <td>102750.000000</td>
      <td>4.650000e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2024.000000</td>
      <td>405000.000000</td>
      <td>2.954083e+06</td>
    </tr>
  </tbody>
</table>
</div>






```python
# Description des variables catégorielles
df.describe(include=['object'])
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
      <th>brand</th>
      <th>model</th>
      <th>fuel_type</th>
      <th>engine</th>
      <th>transmission</th>
      <th>ext_col</th>
      <th>int_col</th>
      <th>accident</th>
      <th>clean_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3269</td>
      <td>3269</td>
      <td>3269</td>
      <td>3269</td>
      <td>3269</td>
      <td>3269</td>
      <td>3269</td>
      <td>3269</td>
      <td>3269</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>52</td>
      <td>1614</td>
      <td>7</td>
      <td>963</td>
      <td>32</td>
      <td>120</td>
      <td>74</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Ford</td>
      <td>M3 Base</td>
      <td>Gasoline</td>
      <td>355.0HP 5.3L 8 Cylinder Engine Gasoline Fuel</td>
      <td>A/T</td>
      <td>Black</td>
      <td>Black</td>
      <td>None reported</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>341</td>
      <td>29</td>
      <td>2815</td>
      <td>48</td>
      <td>939</td>
      <td>802</td>
      <td>1680</td>
      <td>2332</td>
      <td>3269</td>
    </tr>
  </tbody>
</table>
</div>



#### La marque la plus fréquente est Ford


```python
df['brand'].value_counts()
```




    brand
    Ford             341
    BMW              316
    Mercedes-Benz    268
    Chevrolet        259
    Toyota           171
    Porsche          158
    Audi             153
    Lexus            136
    Jeep             114
    Land             100
    Nissan            95
    Cadillac          92
    Dodge             84
    GMC               84
    RAM               72
    Subaru            59
    Hyundai           57
    Mazda             57
    INFINITI          54
    Volkswagen        51
    Honda             49
    Kia               46
    Acura             45
    Lincoln           44
    Jaguar            39
    Volvo             33
    MINI              31
    Maserati          31
    Bentley           27
    Chrysler          25
    Buick             25
    Mitsubishi        20
    Genesis           16
    Hummer            16
    Pontiac           15
    Lamborghini       15
    Alfa              12
    Rolls-Royce       10
    Ferrari            9
    Aston              8
    Scion              6
    Saturn             5
    McLaren            4
    FIAT               4
    Lotus              3
    Mercury            3
    Saab               2
    Bugatti            1
    Plymouth           1
    smart              1
    Maybach            1
    Suzuki             1
    Name: count, dtype: int64



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


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_25_0.png)
    


#### La distribution des prix des voitures est asymétrique à droite. les voitures les moins chers sont les plus nombreuses

### quelques stats interessantes



```python
most_expensive_car = df.loc[df['price'].idxmax()]
print("Voiture la plus chère :")
print(most_expensive_car[['brand', 'model', 'price']])
```

    Voiture la plus chère :
    brand             Maserati
    model    Quattroporte Base
    price            2954083.0
    Name: 693, dtype: object
    


```python
# Voiture la moins chère
cheapest_car = df.loc[df['price'].idxmin()]
print("\nVoiture la moins chère :")
print(cheapest_car[['brand', 'model', 'price']])
```

    
    Voiture la moins chère :
    brand           Lincoln
    model    Aviator Luxury
    price            2000.0
    Name: 425, dtype: object
    


```python
brand_counts = df['brand'].value_counts()
print("\nNombre de voitures par marque :")
print(brand_counts)
```

    
    Nombre de voitures par marque :
    brand
    Ford             341
    BMW              316
    Mercedes-Benz    268
    Chevrolet        259
    Toyota           171
    Porsche          158
    Audi             153
    Lexus            136
    Jeep             114
    Land             100
    Nissan            95
    Cadillac          92
    Dodge             84
    GMC               84
    RAM               72
    Subaru            59
    Hyundai           57
    Mazda             57
    INFINITI          54
    Volkswagen        51
    Honda             49
    Kia               46
    Acura             45
    Lincoln           44
    Jaguar            39
    Volvo             33
    MINI              31
    Maserati          31
    Bentley           27
    Chrysler          25
    Buick             25
    Mitsubishi        20
    Genesis           16
    Hummer            16
    Pontiac           15
    Lamborghini       15
    Alfa              12
    Rolls-Royce       10
    Ferrari            9
    Aston              8
    Scion              6
    Saturn             5
    McLaren            4
    FIAT               4
    Lotus              3
    Mercury            3
    Saab               2
    Bugatti            1
    Plymouth           1
    smart              1
    Maybach            1
    Suzuki             1
    Name: count, dtype: int64
    


```python
average_price_by_brand = df.groupby('brand')['price'].mean().sort_values(ascending=False)
print("\nPrix moyen par marque :")
print(average_price_by_brand)
```

    
    Prix moyen par marque :
    brand
    Bugatti          1.950995e+06
    Rolls-Royce      3.863920e+05
    Lamborghini      2.859089e+05
    Ferrari          2.340876e+05
    McLaren          2.322362e+05
    Maserati         1.416163e+05
    Aston            1.254932e+05
    Bentley          1.228501e+05
    Porsche          8.590918e+04
    Maybach          6.425000e+04
    Lotus            5.841667e+04
    Mercedes-Benz    5.035782e+04
    Land             4.987347e+04
    RAM              4.157493e+04
    Genesis          4.134606e+04
    Cadillac         3.887376e+04
    BMW              3.787408e+04
    Chevrolet        3.627338e+04
    Alfa             3.535750e+04
    GMC              3.521315e+04
    Audi             3.411637e+04
    Ford             3.358521e+04
    Lexus            3.284590e+04
    Dodge            3.276021e+04
    Jeep             2.911087e+04
    Jaguar           2.894518e+04
    Plymouth         2.850000e+04
    Toyota           2.833211e+04
    Nissan           2.649803e+04
    Lincoln          2.592191e+04
    Volvo            2.496885e+04
    Kia              2.486587e+04
    Acura            2.217398e+04
    INFINITI         2.179615e+04
    Subaru           2.157519e+04
    Volkswagen       2.064522e+04
    Hummer           1.961806e+04
    Mazda            1.937935e+04
    Buick            1.890456e+04
    Honda            1.885224e+04
    Hyundai          1.763791e+04
    Mitsubishi       1.755065e+04
    FIAT             1.474975e+04
    Mercury          1.423333e+04
    MINI             1.358719e+04
    Pontiac          1.357293e+04
    Chrysler         1.242568e+04
    Saturn           1.237900e+04
    Saab             1.022500e+04
    Scion            8.832167e+03
    Suzuki           6.900000e+03
    smart            5.000000e+03
    Name: price, dtype: float64
    


```python
accident_counts = df['accident'].value_counts()
print("\nNombre de voitures accidentées vs non accidentées :")
print(accident_counts)
```

    
    Nombre de voitures accidentées vs non accidentées :
    accident
    None reported                             2332
    At least 1 accident or damage reported     937
    Name: count, dtype: int64
    


```python
average_price_by_accident = df.groupby('accident')['price'].mean()
print("\nPrix moyen des voitures accidentées vs non accidentées :")
print(average_price_by_accident)
```

    
    Prix moyen des voitures accidentées vs non accidentées :
    accident
    At least 1 accident or damage reported    28492.493063
    None reported                             46363.540309
    Name: price, dtype: float64
    



### Corrélation entre les variables numériques


```python
# Corrélation entre les variables numériques
correlation_matrix = df[['price','model_year']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation entre les variables numériques')
plt.show()
```


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_36_0.png)
    


Il y a qu'une tres faible correlation positive entre le prix et l'année de fabrication de la voiture


```python
# Analyse des variables catégorielles
categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
df_saved=df[['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title','price','model_year','milage']]
df=df[['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title','price','model_year']]

for col in categorical_columns:
    print(f"\nValeurs uniques pour {col}: {df[col].unique()}")
```

    
    Valeurs uniques pour brand: ['Ford' 'Hyundai' 'INFINITI' 'Audi' 'BMW' 'Lexus' 'Aston' 'Toyota'
     'Lincoln' 'Land' 'Mercedes-Benz' 'Dodge' 'Nissan' 'Jaguar' 'Chevrolet'
     'Kia' 'Jeep' 'Bentley' 'MINI' 'Porsche' 'Hummer' 'Chrysler' 'Acura'
     'Volvo' 'Cadillac' 'Maserati' 'Genesis' 'Volkswagen' 'GMC' 'RAM' 'Subaru'
     'Alfa' 'Ferrari' 'Scion' 'Mitsubishi' 'Mazda' 'Saturn' 'Honda' 'Bugatti'
     'Lamborghini' 'Rolls-Royce' 'McLaren' 'Buick' 'Lotus' 'Pontiac' 'FIAT'
     'Saab' 'Mercury' 'Plymouth' 'smart' 'Maybach' 'Suzuki']
    
    Valeurs uniques pour model: ['Utility Police Interceptor Base' 'Palisade SEL' 'Q50 Hybrid Sport' ...
     'CT 200h Base' 'Impala 2LZ' 'Continental GT Speed']
    
    Valeurs uniques pour fuel_type: ['E85 Flex Fuel' 'Gasoline' 'Hybrid' 'Diesel' 'Plug-In Hybrid' '–'
     'not supported']
    
    Valeurs uniques pour transmission: ['6-Speed A/T' '8-Speed Automatic' '7-Speed A/T' 'A/T' '8-Speed A/T'
     'Transmission w/Dual Shift Mode' '9-Speed Automatic' '6-Speed M/T'
     'Automatic' '10-Speed A/T' '9-Speed A/T' '5-Speed A/T'
     '6-Speed Automatic with Auto-Shift' 'M/T' 'CVT Transmission'
     '4-Speed A/T' '6-Speed Automatic' '4-Speed Automatic' 'Automatic CVT'
     '8-Speed Automatic with Auto-Shift' '7-Speed Automatic with Auto-Shift'
     '5-Speed M/T' '7-Speed Manual' '10-Speed Automatic' '6-Speed Manual'
     'Transmission Overdrive Switch' '7-Speed Automatic' '–'
     '5-Speed Automatic' '7-Speed' '7-Speed M/T' '7-Speed DCT Automatic']
    
    Valeurs uniques pour ext_col: ['Black' 'Moonlight Cloud' 'Blue' 'Green' 'Silver' 'Yellow' 'White' 'Gray'
     'Purple' 'Iconic Silver Metallic' 'Mythos Black Metallic' 'Red' 'Gold'
     'Horizon Blue' 'Orange' 'Beige' 'Summit White' 'Bright White Clearcoat'
     'Crystal Black Silica' 'Ultra Black' 'Lunare White Metallic' 'Hyper Red'
     'Vik Black' 'Sonic Silver Metallic' 'Patriot Blue Pearlcoat'
     'Black Cherry' 'Blu' 'Beluga Black' 'Brown' 'Cobra Beige Metallic'
     'Anodized Blue Metallic' 'Delmonico Red Pearlcoat' '–' 'Black Clearcoat'
     'Machine Gray Metallic' 'Twilight Black' 'Diamond Black' 'Maroon'
     'Firecracker Red Clearcoat' 'Onyx' 'Santorini Black'
     'Mosaic Black Metallic' 'Deep Black Pearl Effect'
     'Polymetal Gray Metallic' 'Jet Black Mica' 'Dark Gray Metallic'
     'Isle of Man Green Metallic' 'Volcano Grey Metallic' 'Redline Red'
     'Sandstone Metallic' 'Tan' 'Silver Zynith' 'Velvet Red Pearlcoat' 'Pink'
     'Black Obsidian' 'Midnight Blue Metallic' 'Matte White'
     'Iridium Metallic' 'Magnetic Black' 'Pacific Blue'
     'White Diamond Tri-Coat' 'Dark Matter Metallic' 'Diamond White' 'Tempest'
     'Tango Red Metallic' 'Majestic Plum Metallic' 'Dark Moon Blue Metallic'
     'Manhattan Noir Metallic' 'Nebula Gray Pearl' 'Onyx Black' 'Frozen White'
     'Shadow Black' 'Sting Gray Clearcoat' 'Maximum Steel Metallic'
     'Carbonized Gray Metallic' 'Atomic Silver' 'Crimson Red Tintcoat'
     'Billet Silver Metallic Clearcoat' 'Iridescent Pearl Tricoat'
     'Nero Daytona' 'Rich Garnet Metallic' 'Nero Noctis'
     'Ebony Twilight Metallic' 'Blue Reflex Mica' 'Eiger Grey'
     'Granite Crystal Clearcoat Metallic' 'Dark Slate Metallic'
     'Agate Black Metallic' 'Stone Gray Metallic' 'Bayside Blue'
     'Silver Ice Metallic' 'Siren Red Tintcoat' 'Shadow Gray Metallic'
     'Satin Steel Metallic' 'Red Quartz Tintcoat' 'Carrara White Metallic'
     'Soul Red Crystal Metallic' 'DB Black Clearcoat'
     'Iridium Silver Metallic' 'White Frost Tri-Coat' 'Glacial White Pearl'
     'Cajun Red Tintcoat' 'Alpine White' 'Ember Pearlcoat'
     'Twilight Blue Metallic' 'Burnished Bronze Metallic'
     'Ice Silver Metallic' 'Hellayella Clearcoat' 'Snowflake White Pearl'
     'China Blue' 'White Knuckle Clearcoat' 'Nautical Blue Pearl'
     'Tungsten Metallic' 'Magnetic Metallic' 'Kinetic Blue'
     'Black Sapphire Metallic' 'Brilliant Silver Metallic'
     'Glacier Silver Metallic' 'Nightfall Gray Metallic' 'C / C']
    
    Valeurs uniques pour int_col: ['Black' 'Gray' 'Green' 'Brown' 'White' '–' 'Beige' 'Jet Black' 'Red'
     'Blue' 'Charcoal' 'Medium Pewter' 'Ice' 'Obsidian Black' 'Orange' 'Nero'
     'Sahara Tan' 'Ebony' 'Hotspur' 'Nougat Brown' 'Navy Pier' 'Grace White'
     'Mesa' 'Medium Dark Slate' 'Gold' 'Charles Blue' 'Portland'
     'Medium Light Camel' 'Black / Saddle' 'Ebony / Pimento'
     'Mistral Gray / Raven' 'Silver' 'Medium Stone' 'Kyalami Orange' 'Boulder'
     'Charcoal Black' 'Dark Auburn' 'Diesel Gray / Black' 'Global Black'
     'Black / Brown' 'Black / Stone Grey' 'Graphite' 'Nero Ade' 'Tan'
     'Saddle Brown' 'Light Titanium' 'Dark Gray' 'Platinum' 'Shale'
     'Black Onyx' 'Sandstone' 'Cobalt Blue' 'Deep Cypress' 'Ceramic'
     'Light Platinum / Jet Black' 'Dark Galvanized' 'Shara Beige' 'Rioja Red'
     'Yellow' 'Light Slate' 'Brandy' 'Titan Black / Quarzit' 'Chestnut'
     'Medium Earth Gray' 'Red / Black' 'Ebony Black' 'Deep Garnet'
     'Ebony / Ebony Accents' 'Sport' 'Medium Ash Gray' 'White / Brown'
     'Hotspur Hide' 'Canberra Beige' 'Deep Chestnut']
    
    Valeurs uniques pour accident: ['At least 1 accident or damage reported' 'None reported']
    
    Valeurs uniques pour clean_title: ['Yes']
    

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


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_40_0.png)
    



    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_40_1.png)
    


## PRETRAITEMENT DES DONNEES 


```python
# Encodage des variables catégorielles avec get_dummies
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Conversion des booléens (True/False) en entiers (1/0)
df = df.astype(int)

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
      <th>price</th>
      <th>model_year</th>
      <th>cluster</th>
      <th>brand_Alfa</th>
      <th>brand_Aston</th>
      <th>brand_Audi</th>
      <th>brand_BMW</th>
      <th>brand_Bentley</th>
      <th>brand_Bugatti</th>
      <th>brand_Buick</th>
      <th>...</th>
      <th>int_col_Shara Beige</th>
      <th>int_col_Silver</th>
      <th>int_col_Sport</th>
      <th>int_col_Tan</th>
      <th>int_col_Titan Black / Quarzit</th>
      <th>int_col_White</th>
      <th>int_col_White / Brown</th>
      <th>int_col_Yellow</th>
      <th>int_col_–</th>
      <th>accident_None reported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10300</td>
      <td>2013</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38005</td>
      <td>2021</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15500</td>
      <td>2015</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31000</td>
      <td>2017</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7300</td>
      <td>2001</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1897 columns</p>
</div>



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

    MSE: 91375408111999405773542129715002736640.00
    R2: -6239874089636566354316230656.00
    MAPE: 28781226957210692.00%
    

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

    === Régression Linéaire ===
    MAE : 26533.04
    MSE : 14127233563.18
    RMSE : 118858.04
    R² : 0.04
    Erreur relative moyenne : 104.74 %
    

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

    
    === Régression Polynomiale (degré 2) ===
    MAE : 24786.71
    MSE : 13995156857.94
    RMSE : 118301.13
    R² : 0.04
    Erreur relative moyenne : 80.04 %
    
    === Régression Polynomiale (degré 3) ===
    MAE : 24287.52
    MSE : 13950298910.98
    RMSE : 118111.38
    R² : 0.05
    Erreur relative moyenne : 77.34 %
    
    === Régression Polynomiale (degré 4) ===
    MAE : 24307.58
    MSE : 13971645290.25
    RMSE : 118201.71
    R² : 0.05
    Erreur relative moyenne : 77.88 %
    


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_50_1.png)
    


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

    === Régression Exponentielle ===
    MAE : 21298.10
    MSE : 14179198606.41
    RMSE : 119076.44
    R² : 0.03
    Erreur relative moyenne : 56.77 %
    


    
![png](ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_files/ANALYSE%20EXPLORATOIRE%20%2C%20REGRESSION%20%2C%20CLASSIFICATION%20USED_CARS%20-%20DONFACK%20PASCAL%20-%20M1-GI_52_1.png)
    


### La regression exponentielle semble etre la meilleure methode pour predire le prix en fonction du kilometrage


```python

```
