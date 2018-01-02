import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
loldf=[['Adj. Close']]
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)


#maths . ceil runds everyhting up to the nerest whole
#math.ceil returns a float int makes it an int
#0.1 predicts ten percent out of the data frame - ten days later
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
#this shifts all the forecast col up , (negatively shifting) so the adj close
#is the one thats ten days into the future
df['label'] = df[forecast_col].shift(-forecast_out)


#in future process both training and testing data together
#drop the label collumn
x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

#this shuffles up the table
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
#making the classifier
clas = LinearRegression()
#'fit' means train
#change pik to 1 to use previously saved model
pik = 0
if pik == 0 :
    clas.fit(x_train,y_train)
    with open('linearregression.pickle','wb') as f:
        pickle.dump(clas, f)
else:
    pickle_in = open('linearregression.pickle','rb')
    clas = pickle.load(pickle_in)


#score means test
accuracy = clas.score(x_test,y_test)

forecast_set = clas.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan
#.iloc (pandas) finds something by index or position
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
oneday = 60*60*24 #(86400)
next_unix = last_unix + oneday

for i in forecast_set:
    #for loops iterated throught the forcast sets the future numbers in the df to nan
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += oneday
    #because the values are indexed by date it finds the date and if there isnt one
    #makes it it also makes all the other collums nan and adds the i
    #which is the forecast makingit just time and forcast.
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
#loc is location 4 is bottom right
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
