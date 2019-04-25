# This is a different way to get prices rather than using the quandl API
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

# Assign graph style
style.use('fivethirtyeight')

# Get price table
start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)
df = web.DataReader('TSLA', 'yahoo', start, end)

# Variables
returns = df['Adj Close'].pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178

# print(df.head())
# print("\n")
# print(df.tail())
# # df['Adj Close'].plot()
# # plt.show()

# df['100MA'] = df['Adj Close'].rolloing(window=100).mean()
#Removes all rows where 100MA is Nan
# df.dropna(inplace=True)

# rolloing(window=100, min_periods=0).mean will start averaging with no minimum requirement. I.e. averaging n and n+1 up to n+100 and holding the window at that point.
# df['100MA'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

# #making some graphs
# ax1=plt.subplot2grid((6,1),(0,0),rowspan=5, colspan=1)
# ax2=plt.subplot2grid((6,1),(5,0),rowspan=5, colspan=1, sharex=ax1)

# ax1.plot(df.index,df['Adj Close'])
# ax1.plot(df.index,df['100MA'])
# ax2.bar(df.index,df['Volume'])

# plt.show()

ax1=plt.subplot2grid((6,1),(0,0),rowspan=5, colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=5, colspan=1, sharex=ax1)