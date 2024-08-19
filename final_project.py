import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
import pandas as pd
import numpy as np

def main():
    # Loads data and extracts risk-free rate
    df = pd.read_csv("Data.CSV", skiprows=3, parse_dates=[0], index_col=0)
    riskfree = df.loc["2011":"2024-01-01","RF"]

    # Gets stock and index input from user
    stock, index = get_input()

    # Gets the adjusted close price of both stock and index
    stock_prices = get_data(stock)
    index_prices = get_data(index)
    
    # Calculates the stock and index risk premium using their respective prices and the risk-free rate
    index_pr = get_excess(index_prices, riskfree)
    stock_pr = get_excess(stock_prices, riskfree)

    # Displays regression results
    print(regress_stats(index_pr, stock_pr))
    regress(index_pr, stock_pr).show()

# Takes in inputs for both stock ticker and index ticker
def get_input():
    stock = input("Enter the stock ticker: ").upper()
    index = input("Enter the index ticker: ").upper()
    return stock, index

# Collects historical price data for both the stock and the index
def get_data(ticker):
    if ticker.isalnum() and len(ticker)<=5:
        prices = yf.download(ticker, start="2011-01-01", end="2024-01-01")
        if prices.empty:
            raise ValueError(f"No data found for {ticker}")
        else:
           return prices["Adj Close"]
    else:
        raise ValueError("Invalid ticker symbol")

# Calculates risk premium given prices and risk-free rate
def get_excess(prices,riskfree):
    returns = prices.pct_change()
    excess = (returns-riskfree)*100
    return excess.dropna()

# Plots a regression line and scatterplot
def regress(x, y):
    plt.scatter(x, y, label="Data")
    a,b = np.polyfit(x, y, deg=1)
    y_pred = a*x+b
    plt.plot(x, y_pred, label=f"Fitted Line: Y = {a:.2f}X + {b:.2f}", color="black",linewidth=2)
    plt.title("Linear regression of excess stock returns on excess index returns")
    plt.xlabel("Excess index returns (%)")
    plt.ylabel("Excess stock returns (%)")
    plt.legend()
    plt.grid()
    return plt

# Displays a regression table containing key statistics
def regress_stats(x, y):
    x = sm.add_constant(x)
    mod = sm.OLS(y, x)
    results = mod.fit(cov_type="HC3")
    return results.summary()

if __name__ == "__main__":
    main()
