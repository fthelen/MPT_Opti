# The goal of this script is to obtain weekly price data for n number of securities

# Import Pandas web data access library
from pandas_datareader import data as web

# Request quote information for single ticker

singleQuote = web.DataReader('TSLA', data_source='yahoo', start='2010-01-01', end='2014-12-31')

print(singleQuote) 