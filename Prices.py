# The goal of this script is to obtain weekly price data for n number of securities

# For math and graphs later
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import scipy.optimize as sco
import seaborn as sns

# Import list of tickers to get prices (using static list for now)
stocks = ['AAPL', 'MSFT','KO','PEP','GOOGL','VZ']

# Add in API key
quandl.ApiConfig.api_key = 'yy5biiJzFSehT4GjgxDN'

# Call API for data
data = quandl.get_table('WIKI/PRICES', ticker = stocks, 
    qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
    date = { 'gte': start, 'lte': end }, paginate=True)
data.head()

df = data.set_index('date')
table = df.pivot(columns='ticker')
# By specifying col[1] in below list comprehension
# You can select the stock names under multi-level column
table.columns = [col[1] for col in table.columns]
table.head()