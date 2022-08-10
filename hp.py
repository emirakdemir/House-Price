# housePrice prediction #


import sklearn
import seaborn as sns
import matplotlib.mlab as mlab 
import torch
    
############ LIBRARIES ############

# BASE
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt

# DATA PREPROCESSING
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor 

# MODELING
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# MODEL TUNING
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# WARNINGS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


path1 = 'train.csv'
path2 = 'test.csv'

df_train = pd.read_csv(path1)
df_train.head()

df_test = pd.read_csv(path2)
df_test.head()

df_train.shape
df_test.shape


# TRAIN ve TEST SETINI BIRLESTIRME
df = df_train.append(df_test).reset_index(drop=True)
df.shape

# Birleştirdiğimiz test veri setinde SalePrice sütunu bulunmuyordu, bu değerlere NaN ataması yapıyor, kaç tane olduğunu yeniden kontrol etmek istersek nasıl bakabiliriz?
df.value_counts(np.where(df["SalePrice"]>0,'1','0'))


# EXPLORATORY DATA ANALYSIS

def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat


cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


df[num_cols].describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99]).T



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("         ")


for col in num_cols:
    num_summary(df, col)


# BAĞIMLI DEĞİŞKENİN İNCELENMESİ

df["SalePrice"].describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).T

sns.set(rc={'figure.figsize': (6, 6)})
df["SalePrice"].hist(bins=100)
plt.show()


df=df.loc[~(df.SalePrice>600000 ),]
df["SalePrice"].hist(bins=100)
plt.show()

("Çarpıklık: %f" % df['SalePrice'].skew()) #1.42

np.log1p(df['SalePrice']).hist(bins=50)
plt.show()

("Çarpıklık: %f" % np.log1p(df['SalePrice']).skew()) #0.029

df.head()



# KORELASYON

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdPu")
plt.show(block=True)


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=False)



# AYKIRI GÖZLEM

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



for col in num_cols:
    if col != "SalePrice":
      print(col, check_outlier(df, col))


# EKSIK DEGER ANALIZI


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]


# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
for col in no_cols:
    df[col].fillna("No",inplace=True)

missing_values_table(df)

df.shape


# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)



# FEATURE ENGINEERING
#   Rare yapılacaklar: MSZoning, LotShape, ExterCond, GarageQual, BsmtFinType2 , Condition1 , BldgType 
#   Çıkartılacaklar: Street, Alley, LandContour, Utilities, LandSlope, Condition2, Heating, CentralAir, Functional, PoolQC, MiscFeature, Neighborhood, KitchenAbvGr

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)


# MSZoning: aşam alanının Zone'unu belirtmektedir. Residential High grubu az olduğu için ile Residential Medium ile birleştirebiliriz. Diğer iki grubun sayıları düşük olduğu için daha anlamlı hale gelebilmesi adına bir araya getirebiliriz.
df["MSZoning"].value_counts()


df.loc[(df["MSZoning"] == "RH"), "MSZoning"] = "RM"
df.loc[(df["MSZoning"] == "FV"), "MSZoning"] = "FV + C (all)"
df.loc[(df["MSZoning"] == "C (all)"), "MSZoning"] = "FV + C (all)"
df["MSZoning"].value_counts()


# Lot Area: Evin ft2'sini göstermektedir. 200K ya kadar değer vardır ancak çoğunluk alt değerlerde olduğundan bizim için anlam yaratabilmesi için gruplandırabiliriz.
sns.set(rc={'figure.figsize': (5, 5)})
bins = 50
plt.hist(df["LotArea"],bins, alpha=0.5, density=True)
plt.show()

df["LotArea"].max()
df['LotArea'].mean()



New_LotArea =  pd.Series(["Studio","Small", "Middle", "Large","Dublex","Luxury"], dtype = "category")
df["New_LotArea"] = New_LotArea
df.loc[(df["LotArea"] <= 2000), "New_LotArea"] = New_LotArea[0]
df.loc[(df["LotArea"] > 2000) & (df["LotArea"] <= 4000), "New_LotArea"] = New_LotArea[1]
df.loc[(df["LotArea"] > 4000) & (df["LotArea"] <= 6000), "New_LotArea"] = New_LotArea[2]
df.loc[(df["LotArea"] > 6000) & (df["LotArea"] <= 8000), "New_LotArea"] = New_LotArea[3]
df.loc[(df["LotArea"] > 10000) & (df["LotArea"] <= 12000), "New_LotArea"] = New_LotArea[4]
df.loc[df["LotArea"] > 12000 ,"New_LotArea"] = New_LotArea[5]

df["New_LotArea"].value_counts()


# LotShape: Mülkün genel şeklini göstermektedir. 4 tane grubu olmasından da bizim için reg ve IR olarak iki grubu olması yeterlidir.
df["LotShape"].value_counts()

df.loc[(df["LotShape"] == "IR1"), "LotShape"] = "IR"
df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR"

df["LotShape"].value_counts()


# ExterCond: Malzemenin dış cephedeki durumunu vermektedir.
df["ExterCond"].value_counts()

df["ExterCond"] = np.where(df.ExterCond.isin(["Fa", "Po"]), "FaPo", df["ExterCond"])
df["ExterCond"] = np.where(df.ExterCond.isin(["Ex", "Gd"]), "ExGd", df["ExterCond"])
df['ExterCond'].value_counts()


# GarageQual: Garajın kalitesi.
df['GarageQual'].value_counts()

df["GarageQual"] = np.where(df.GarageQual.isin(["Fa", "Po"]), "FaPo", df["GarageQual"])
df["GarageQual"] = np.where(df.GarageQual.isin(["Ex", "Gd"]), "ExGd", df["GarageQual"])
df["GarageQual"] = np.where(df.GarageQual.isin(["ExGd", "TA"]), "ExGd", df["GarageQual"])
df['GarageQual'].value_counts()


# BsmtFinType1 ve BsmtFinType2: Birinci ve ikinci bodrumun bitmiş bölümünün kalitesi.
df['BsmtFinType1'].value_counts()

df['BsmtFinType2'].value_counts()


df["BsmtFinType1"] = np.where(df.BsmtFinType1.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType1"])
df["BsmtFinType1"] = np.where(df.BsmtFinType1.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType1"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType2"])
df['BsmtFinType1'].value_counts()

df['BsmtFinType2'].value_counts()


# Condition1: Anayola ya da demiryoluna yakınlığını gösterir.
df['Condition1'].value_counts()

df.loc[(df["Condition1"] == "Feedr") | (df["Condition1"] == "Artery") |(df["Condition1"] == "RRAn") |(df["Condition1"] == "PosA") | (df["Condition1"] == "RRAe"),"Condition1"] = "AdjacentCondition"
df.loc[(df["Condition1"] == "RRNn") | (df["Condition1"] == "PosN") |(df["Condition1"] == "RRNe"),"Condition1"] = "WithinCondition"
df.loc[(df["Condition1"] == "Norm") ,"Condition1"] = "NormalCondition"
df['Condition1'].value_counts()


# Condition2: İkinci yol varsa onnu göstermektedir.
df['Condition2'].value_counts()

df.drop('Condition2',axis=1,inplace=True)


# BldgType: Binanın türünü vermektedir.
df['BldgType'].value_counts()

df["BldgType"] = np.where(df.BldgType.isin(["1Fam", "2fmCon"]), "Normal", df["BldgType"])
df["BldgType"] = np.where(df.BldgType.isin(["TwnhsE", "Twnhs", "Duplex"]), "Big", df["BldgType"])
df['BldgType'].value_counts()


# TotalQual: Kaliteyi gösteren değişkenlerle toplam bir kalite göstergesi değişkeni oluşturalım
df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1)


# Overall: Evin genel durumu ve kullanılan materyallerin kalitesinden bir değişken oluşturalım.
df["Overall"] = df[["OverallQual", "OverallCond"]].sum(axis = 1)


# NEW_TotalFlrSF: Evin toplamdaki kapladığı yüzey alanı
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]


# NEW_TotalBsmtFin: Tamamlanmış toplam bodrum alanı
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1+df.BsmtFinSF2


# NEW_PorchAre: Ev dışında kalan toplam alan
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF


# NEW_TotalHouseArea: Evin toplam alanı
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF


# NEW_TotalSqFeet: Evin toplam kapladığı ft2
df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF


# NEW_TotalFullBath ve NEW_TotalHalfBath: Ev içerisindeki yarım ve tam banyo sayıları
df["NEW_TotalFullBath"] = df.BsmtFullBath + df.FullBath
df["NEW_TotalHalfBath"] = df.BsmtHalfBath + df.HalfBath


# NEW_TotalBath: Ev içerisinde bulunan toplam banyo sayısını ifade etmektedir.
df["NEW_TotalBath"] = df["NEW_TotalFullBath"] + (df["NEW_TotalHalfBath"]*0.5)


# LotRatio: Yaşanan alan, toplam ev alanı ve garaj alanının arazinin ne kadarını kapladığı
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea
df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea
df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea


# Tarihler arasındaki farklar: Restorasyon ile yapım yılı arasında geçen yıl, Garajın yapım yılıyla evin yapım yılı arasındaki fark gibi değişkenler
df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt
df["NEW_HouseAge"] = df.YrSold - df.YearBuilt
df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd
df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt
df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)
df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt



df.head()

drop_list = ["Street", "Alley", "LandContour", "Utilities" ,"LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood","KitchenAbvGr", "CentralAir", "Functional"]
df.drop(drop_list, axis=1, inplace=True)



# Encoding
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)



# Modelleme #
missing_values_table(df)

# Log dönüşümünün gerçekleştirilmesi
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

y = np.log1p(df[df['SalePrice'].notnull()]['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    
    
xgboost_model = XGBRegressor(objective='reg:squarederror')

rmse = np.mean(np.sqrt(-cross_val_score(xgboost_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))


xgboost_params = {"learning_rate": [0.1, 0.01, 0.03],
                  "max_depth": [5, 6, 8],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1]}

xgboost_gs_best = GridSearchCV(xgboost_model,
                            xgboost_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

xgboost_gs_best.best_params_

final_model = xgboost_model.set_params(**xgboost_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

round(rmse, 4)



# Modelin Test Edilmesi
xgboost_tuned = XGBRegressor(objective='reg:squarederror',**xgboost_gs_best.best_params_).fit(X_train, y_train)
y_pred = xgboost_tuned.predict(X_test)


# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
new_y= np.expm1(y_pred)
new_y_test= np.expm1(y_test)

np.sqrt(mean_squared_error(new_y_test, new_y))
# RMSE : 23535.96597150668

df['SalePrice'].mean()

test_df.head()