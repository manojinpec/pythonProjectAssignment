
import pandas as pd
import numpy as np
import re
import requests
import mplfinance as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Function to fetch latest price from an API Call
def Watchlist(Stock):
   for i in Stock:
       print(i)
       urlPart1 = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='
       urlPart2 = '{}&outputsize=full&apikey=1T72KQVIWEZ3JT70'.format(i)
       url = urlPart1 + urlPart2
       print(url)
       data = requests.get(url)
       print(data)
       # Converting the data into a Dictionary which has two keys 'Meta' and 'Time Series (Daily)'
       dict1 = data.json()
       print(type(dict1))
       #print(dict1.keys())
       dict2 = dict1['Time Series (Daily)']
       #print(type(dict2))
       df_API = pd.DataFrame.from_dict(dict2)
       # Transpose the data frame to have dates as rows and other fields as columns
       df_API_T = df_API.transpose()
       df_DailyData = df_API_T.reset_index()
       df_DailyData.rename(columns={'index': 'Date',
                                    '1. open': 'open', '2. high': 'high',
                                    '3. low': 'low', '4. close': 'close',
                                    '5. volume': 'volume'}, inplace=True)

# Function to convert an Object to Datetime and extracting Year of each date
def DateConverter(Date):
    Date1 = pd.to_datetime(Date)
    Year1 = pd.DatetimeIndex(Date1).year
    return(Date1,Year1)

Stock = ['IBM','ACC','INFO']
Watchlist(Stock)
df_Prices = pd.read_csv('prices.csv')
df_Securities = pd.read_csv('securities.csv')

# Understand the data set for Prices
df_Prices.info()

# Understand the data for Securities detail
df_Securities.info()

# Re-naming the 'Ticker' in Securities data to 'symbol' for merging
df_Securities.rename(columns={'Ticker symbol': 'symbol',
                              'Date first added': 'Inception Date',
                              'Security': 'Company Name'},
                     inplace=True)

df_merged = pd.merge(df_Securities, df_Prices, on='symbol', how='outer')
df_merged.info()
print(df_merged.head())

# Understanding the Merged data
df_merged.info()
print(df_merged.head())
print(df_merged.notnull().count()) # Getting Not Null count

df_Securities.info()
V1 = []
V1 = list(df_Securities['Inception Date']) # To be used for visualisation

# Check for Not Null values and replace with NA
df_Securities['Inception Date'] = np.where(df_Securities['Inception Date'].isnull(), 'NA', df_Securities['Inception Date'])

# Count of Inception Date now shows 505 as other fields
df_Securities.info()
print(df_Securities['Inception Date'].value_counts().sort_index())

Inception_Date, year = DateConverter(V1) # Call of a function to covert into Datetime object
VYear = year.value_counts()
x = list(VYear.index)
y = list(VYear)

fig, ax = plt.subplots()
width = 0.75 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="gray")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
for i, v in enumerate(y):
    ax.text(v + .25, i + .25, str(v), color='gray', fontweight='bold') #add value labels into bar
plt.title('No. of Companies vs Launched Year')
plt.xlabel('No. of Companies')
plt.ylabel('Launch Year')
plt.show()

# Using Securities data 'Address of Headquarters' and fetch the City for the same
R1 = r"\w+\s?\w*$"
C = []
for i in range(len(df_Securities['Address of Headquarters'])):
    S = str(df_Securities['Address of Headquarters'][i])
    C.append(re.findall(R1, S))
df_Securities['City'] = C
print(df_Securities.head(5))
df_Securities.info()


# Regression Algo
# Filtering the Prices dataframe on a particular symbol for IBMe : IBM
selected_symbol = ['IBM']
df_Prices_IBM = df_Prices[df_Prices['symbol'].isin(selected_symbol)]
df_Prices_IBM.info()

df_Prices_IBM['date']= pd.to_datetime(df_Prices_IBM['date'])

print(df_Prices_IBM.dtypes)

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encoding Dates to be unique by passing Label_encoder
df_Prices_IBM['date'] = label_encoder.fit_transform(df_Prices_IBM['date'])
print(df_Prices_IBM['date'].unique())
df_Prices_IBM['date'].apply(lambda x: float(x))

df_Prices_IBM1 = df_Prices_IBM[['date','open','close','low','high','volume']]
print('---------Test New Name -------------------------')
print(df_Prices_IBM1.head())

print('Printing the Co-Relation Matrix')
corrMetrics = df_Prices_IBM1.corr()
print(corrMetrics)

# Visualize the Correlation Heat map with Seaborn
top_corr_features = corrMetrics.index
plt.figure(figsize=(20,20))
g = sns.heatmap(df_Prices_IBM1[top_corr_features].corr(), annot=True, cmap="RdYlGn")


#With Co-relation metrics its evident that Volumn is least corelated with any other feature.

X = df_Prices_IBM1.drop('volume', axis=1).values  # Feature
y = df_Prices_IBM1['volume'].values  # Target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using Lasso for identifying most important predictor.
names = df_Prices_IBM1.drop('volume', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

# Using Lasso Regression for Regularize
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print('Lasso Regression score is: ',lasso.score(X_test, y_test))


# Hyperparameter tuning for Lasso Regression
alpha = [0.001, 0.01, 0.1, 1]
param_Lasso1 = dict(alpha=alpha)
grid_lasso = GridSearchCV(estimator=lasso, param_grid=param_Lasso1, scoring='r2', verbose=1, n_jobs=-1, cv=10)
grid_Lasso_result = grid_lasso.fit(X_train, y_train)

print('Best Lasso Params grid_Lasso_result: ', grid_Lasso_result.best_params_)
print('Best Lasso Score grid_Lasso_result: ', grid_Lasso_result.best_score_)



# Using Ridge Regression for Regularize
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge Regression Score is :", ridge.score(X_test, y_test))


# Hyperparameter tuning for Ridge Regression
alpha = [0.001, 0.01, 0.1, 1]
param_grid = dict(alpha=alpha)
grid = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1, cv=10)
grid_result = grid.fit(X_train, y_train)
print('Best Params for Ridge Regression: ', grid_result.best_params_)
print('Best Score for Ridge Regression : ', grid_result.best_score_)
