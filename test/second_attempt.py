import bs4 as bs
import datetime as dt
import os
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import pickle
import requests
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
from time import sleep
from random import randint
import scipy.optimize as sco
import numpy as np

# date time
date_today = dt.date.today()

# All data extraction functions
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    
    tickers = []    
    
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.','-').strip()
        tickers.append(ticker)
    
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)
   
    return tickers

def get_data_from_yahoo(reload_sp500=False):
    nowork = []
    yf.pdr_override()
    
    if not os.path.exists(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\sp500tickers.pickle'):
        save_sp500_tickers()
    
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\stock_dfs'):
        os.makedirs('stock_dfs')
    
    for ticker in tqdm(tickers):
                
        if not os.path.exists(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\stock_dfs''/{} {}.csv'.format(ticker, date_today)):
            df = pdr.get_data_yahoo(ticker, period='5y', interval='1mo')
            df.to_csv(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\stock_dfs''/{} {}.csv'.format(ticker, date_today))
            if os.path.getsize(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\stock_dfs''/{} {}.csv'.format(ticker, date_today)) == 1:
                nowork.append(ticker)       
            sleep(randint(1,5))
        else:
            print("Already have {} {}".format(ticker, date_today))
    
    print(nowork)

def compile_data():
    if not os.path.exists(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\sp_500_join_closes.csv'): 
        with open("sp500tickers.pickle","rb") as f:
                tickers = pickle.load(f)
    
        main_df = pd.DataFrame()

        for count,ticker in tqdm(enumerate(tickers)): 
            df = pd.read_csv(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\stock_dfs''/{} {}.csv'.format(ticker, date_today))
            df.set_index('Date', inplace=True)
            df.rename(columns = {'Adj Close' : ticker}, inplace=True)
            df.drop(['Open','High', 'Low', 'Close', 'Volume'], 1, inplace=True)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            main_df.to_csv('sp_500_join_closes.csv')

# Run data functions
get_data_from_yahoo()
compile_data()

# Portfolio optimization and dislay functions
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (min_weight, max_weight)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (min_weight,max_weight)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[0]

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
  
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in tqdm(range(num_portfolios)):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("annualized Return:", round(rp,2))
    print("annualized Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("annualized Return:", round(rp_min,2))
    print("annualized Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(12, 9))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='viridis_r', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualized volatility')
    plt.ylabel('annualized returns')
    plt.legend(labelspacing=0.8)    

# Call up compiled data
table = pd.read_csv(r'C:\Users\FEED\Documents\GitHub\MPT_Opti\sp_500_join_closes.csv')
table.set_index('Date', inplace=True)

# Weight bounds
min_weight = 0.0
max_weight = 0.06

# Variables
returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178

# # Run optimization and show graph
tqdm(display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate))
plt.show()