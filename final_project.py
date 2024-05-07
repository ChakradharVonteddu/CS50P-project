import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import csv
import statsmodels.api as sm

# Takes in inputs for both stock ticker and index ticker
def main():
    stock=(input("Enter the stock ticker: ")).upper()
    index=(input("Enter the index ticker: ")).upper()
    stock_excess=excess_returns(data(stock))
    index_excess=excess_returns(data(index))
    print(str(regress_stats(index_excess,stock_excess)))
    regress(index_excess,stock_excess).show()

# Collects historical price data for both the stock and the index
def data(ticker):
    if ticker.isalnum() and len(ticker)<=5:
        prices=yf.download(ticker, start="2011-01-01",end="2024-01-01")
        return list(prices["Adj Close"])
    else:
        raise ValueError("Invalid Ticker")

# Calculates excess returns of both stock and index
def excess_returns(prices):
    riskfree_rate=[]
    excess=[]
    with open("F-F_Research_Data_5_Factors_2x3_daily.csv") as file:
        reader=csv.reader(file)
        for row in reader:
            if 2011<=int(row[0][0:4])<=2023:
                riskfree_rate.append(row[6])
    for i in range(0,len(prices)-1):
        excess_i=((prices[i+1]-prices[i])/prices[i])*100-float(riskfree_rate[i+1])
        excess.append(excess_i)
    return excess

# Plots a regression line and scatterplot
def regress(x,y):
    plt.scatter(x,y,label="Data")
    results=stats.linregress(x,y)
    y_pred=[]
    for i in range(0,len(x)):
        y_pred.append(results.slope*x[i]+results.intercept)
    plt.plot(x,y_pred,label=f"Fitted Line:Y={results.slope:.2f}*X+{results.intercept:.2f}",color="green")
    plt.legend()
    plt.title("Linear regression of excess stock returns on excess index returns")
    plt.xlabel("Excess index returns(%)")
    plt.ylabel("Excess stock returns(%)")
    plt.grid()
    return plt

# Displays a regression table containing key statistics
def regress_stats(x,y):
    x=sm.add_constant(x)
    mod=sm.OLS(y,x)
    results=mod.fit(cov_type="HC3")
    return results.summary()

if __name__ == "__main__":
    main()
