# Predict-Car-price

# Import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import Dataframes and concat
bmw = pd.read_csv("bmw.csv")
bmw["type"] = "bmw"

hyundi = pd.read_csv("hyundi.csv")
hyundi["type"] = "hyundi"
hyundi.rename(columns = {"tax(£)": "tax"}, inplace = True)

audi = pd.read_csv("audi.csv")
audi["type"] = "audi"

ford = pd.read_csv("ford.csv")
ford["type"] = "ford"


toyota = pd.read_csv("toyota.csv")
toyota["type"] = "toyota"

vw = pd.read_csv("vw.csv")
vw["type"] = "vw"

merc = pd.read_csv("merc.csv")
merc["type"] = "merc"


skoda = pd.read_csv("skoda.csv")
skoda["type"] = "skoda"


df = pd.concat([bmw, audi, ford, hyundi, toyota, vw, merc, skoda], axis=0)


# Dropping ouliers and EDA first run
drop_lst = df.loc[df["mileage"] > 200000].index

df.drop(drop_lst, inplace = True)

drop_lst2 = df.loc[df["mileage"] < 1000].index

df.drop(index = drop_lst2, inplace = True)

drop_price = df.loc[df["price"] > 120000].index

df.drop(drop_price, inplace = True)

# Plotning EDA
fig, ax = plt.subplots(2,2, figsize=(15,10))
sns.histplot(df.price, kde=True, ax=ax[0,0])
sns.countplot(x='transmission', data=df, hue='fuelType', ax=ax[0,1])
sns.lineplot(x='year', y='price', data=df[df.year > 1995], ci=None, ax=ax[1,0])
sns.scatterplot(x='mileage', y='price', data=df, ax=ax[1,1]);

![image](https://user-images.githubusercontent.com/82596274/121549822-7bc82800-ca0e-11eb-9cf7-41951a980b68.png)

# Dropping outliers and EDA secound run

df = 63009 rows

df.loc[df.price > 38000] = 2244 rows

df.loc[df.mileage > 73000] = 2411 rows

df.loc[df.year < 2015].index = tog bort 2 superoutliers en som var byggd 1960 och en som var bygg 2060

df.loc[df.engineSize > 3] = 96 rows 40k plus som är mindre än 3

df = 41602 rows

# Dropping outliers
df.loc[df.year < 1980].index
df.loc[df.year > 2020].index
df.drop(index = 17726, inplace = True)
df.drop(index = 12072, inplace = True)

### Code for dropping ouliers in specifik columns

drop_box_engineSize = df.loc[df.engineSize > 3].index
df.drop(drop_box_engineSize, inplace = True)
drop_box_price = df.loc[df.price > 38000].index
df.drop(drop_box_price, inplace = True)
drop_box_mileage = df.loc[df.mileage > 73000].index
df.drop(drop_box_mileage, inplace = True)

## Plotning EDA secound run
fig, ax = plt.subplots(2,2, figsize=(15,10))
sns.histplot(df.price, kde=True, ax=ax[0,0])
sns.countplot(x='transmission', data=df, hue='fuelType', ax=ax[0,1])
sns.lineplot(x='year', y='price', data=df[df.year > 1995], ci=None, ax=ax[1,0])
sns.scatterplot(x='mileage', y='price', data=df, ax=ax[1,1]);
sns.histplot(df.price, kde=True, ax=ax[0,0])
![image](https://user-images.githubusercontent.com/82596274/121550265-e1b4af80-ca0e-11eb-914d-95a4e21a1277.png)

## X & y and train test split
target_var = "price"
X = df.drop(columns = target_var, axis = 1 )

y = df[target_var]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
   
## Pipeline for Preprocessing the data
from sklearn.pipeline import make_pipeline

## Sklearn for preprocessing the data
#impute missing values (NaNs)
from sklearn.impute import SimpleImputer

# scaling 
from sklearn.preprocessing import StandardScaler

# onehotencoder, creates dummies (compare to pd.get_dummies)
from sklearn.preprocessing import OneHotEncoder

## preprocessing for categorical columns

cat_trans = make_pipeline(SimpleImputer(strategy = "constant", fill_value = "missing"),
             OneHotEncoder(handle_unknown = "ignore"))
             
## Preprocessing for numerical columns

num_trans = make_pipeline(SimpleImputer(strategy = "median"),
             StandardScaler())
num_cols = X.select_dtypes("number").columns
cat_cols = X.select_dtypes("object").columns

# ## Make column transformer
from sklearn.compose import make_column_transformer
preprocessor = make_column_transformer(
(num_trans, num_cols),
(cat_trans, cat_cols))
X_train = preprocessor.transform(X_train)

### Model selection
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor


from sklearn.ensemble import RandomForestRegressor


from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

regressors = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    KNeighborsRegressor(),
]

from sklearn.model_selection import cross_val_score

for i in regressors:
    cv_result = cross_val_score(i, X_train, y_train, cv = 5)
    
    print(i, cv_result)
    print(i, cv_result.mean())
    print("")
    
    
# RandomForestRegressor gridsearch with hyperparameter
from sklearn.model_selection import GridSearchCV


grid = GridSearchCV(RandomForestRegressor(max_depth=10, random_state=10,n_estimators = 100), param_grid = {}, cv = 5)

grid.fit(X_train, y_train)

# Score test
preprocessor.transform(X_test)
best_model.score(preprocessor.transform(X_test), y_test)

output = 0.91058212300758

