# Tyler Chaput
# MIS110 Fall 2023
# Final Project
# I pledge my honor that I have abided by the Stevens Honor System. - TC

# Load packages
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date

# Set random seed for simulation 
np.random.seed(8)

# Get stock tickers ueed in portfolio 
symbols = ['NVDA', 'MELI', 'CRWD', 'BABA']

# Number of stocks in portfolio
num_symbols = len(symbols)

# Specify start and end date for price history
start = dt.datetime(2023,1,1)
end = dt.datetime(2023,11,30)

# Get historicals and setup dataframe
price_df = yf.download(symbols, start, end)['Adj Close']

# Calculate log returns 
returns = np.log(1 + price_df.pct_change())

# Calculate mean returns
mean_returns = returns.mean()

# Create covarience matrix
cov_matrix = returns.cov()

# Set number of portfolios iterations to run in simulation
num_portfolios = 100000

# Set risk free rate 
rfr = 0.02

# Generate Efficient Frontier
def random_portfolios(num_portfolios, mean_returns, cov_matrix, rfr):
    """This function generates random portfolio iterations
    and records the annual performance and allocations of each"""
    results = np.zeros((3, num_portfolios)) # Create array to store results from simulation
    weights_s = [] # Create list to store weights from simulation
    for i in range(num_portfolios):
        weights = np.random.random(num_symbols)
        weights /= np.sum(weights)
        weights_s.append(weights)
        portfolio_std, portfolio_return = p_ann_perform(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - rfr) / portfolio_std
    return results, weights_s

def p_ann_perform(weights, mean_returns, cov_matrix):
    """This function calculates the annual performance
    (returns, standard deviation) of a given portfolio"""
    returns = np.sum(mean_returns * weights) * 252 # Calculate annualized returns
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Calculate annualized standard deviation
    return std, returns

def efficient_frontier(mean_returns, cov_matrix, num_portfolios, rfr):
    """This function takes the max sharpe ratio and min volatility
    portfolios generated in the monte carlo simulation and plots them
    along with their respective allocations and plots the overall
    efficient frontier"""
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, rfr)
    max_sharpe = np.argmax(results[2])
    global rp
    sdp, rp = results[0, max_sharpe], results[1, max_sharpe]
    global max_sharpe_a
    max_sharpe_a = pd.DataFrame(weights[max_sharpe], index = price_df.columns, columns = ['Allocation'])
    max_sharpe_a.Allocation = [round(i * 100, 2) for i in max_sharpe_a.Allocation]
    max_sharpe_a = max_sharpe_a.T

    min_vol = np.argmin(results[0])
    global rp_min
    sdp_min, rp_min = results[0, min_vol], results[1, min_vol]
    global min_vol_a
    min_vol_a = pd.DataFrame(weights[min_vol], index = price_df.columns, columns = ['Allocation'])
    min_vol_a.Allocation = [round(i * 100, 2) for i in min_vol_a.Allocation]
    min_vol_a = min_vol_a.T

    # Print results of Max Sharpe Ratio and Min Volatility Portfolios from Monte Carlo simulation
    print('-' * 80)
    print('Max Sharpe Ratio Portfolio Allocation\n')
    print('Annualized Return:', round(rp, 2))
    print("Annualized Volatility", round(sdp, 2))
    print('\n')
    print(max_sharpe_a)
    print('-' * 80)
    print('Min Volatility Portfolio Allocation\n')
    print('Annualized Return:', round(rp_min, 2))
    print("Annualized Volatility", round(sdp_min, 2))
    print('\n')
    print(min_vol_a)

    # Plot scatter plot of portfolios generated in simulation to yeild efficient frontier
    plt.figure(figsize=(20,10))
    plt.scatter(results[0, :], results[1, :], c = results[2, :], cmap = 'viridis', marker = 'o', s = 10, alpha = 0.5)
    plt.colorbar(label = 'Sharpe Ratio')
    plt.scatter(sdp, rp, marker = 'D', color = 'blue', s = 200, label = 'Max Sharpe Ratio')
    plt.scatter(sdp_min, rp_min, marker = 'D', color = 'red', s = 200, label = 'Min Volatility')
    plt.title('Simulated Portfolio Optimization based on Effiicient Frontier')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Annualized Returns')
    plt.legend(labelspacing = 1.5)

efficient_frontier(mean_returns, cov_matrix, num_portfolios, rfr)

