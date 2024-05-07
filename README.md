# Estimating the coefficients of the capital asset pricing model using linear regression
### Description:
#### I. Overview of the CAPM model: <br>
The Capital-asset pricing model(CAPM) is a relationship between the risk premium of a specific asset and the risk premium of the broader market for the asset's class. The parameters in this model include the asset's beta (measure of its relative volatility),the risk-free rate i.e. the US treasury bill rate, the specific asset's returns and the market's returns. We calculate the risk premium by subtracting the risk free rate from the expected returns. Mathematically, this model can be represented as: <br>
                                                <p align="center"> $E(R_i)-R_f=\beta_i*(E(R_m)-R_f)$ <br> </p>
Here, $E(R_i)$ refers to the expected returns on a particular investment, $R_f$ represents the risk-free rate, $E(R_m)$ refers to the expected market returns and $\beta_i$ is the beta for the particular investment. The $\beta_i$ for a particular investment can be calculated using a time-series linear regression model based on the CAPM utilizing historical data for asset and market returns. The model is as follows: <br>
                                         <p align="center> $R_{i,t}-R_f=\alpha_i+\beta_i*(R_{m,t}-R_f)+\epsilon_{i,t}$ </p> <br>
where $\alpha_i$ and $\beta_i$ are coefficients to be estimated and $\epsilon_{i,t}$ represents the error term. Market returns, asset returns and the error term are time-indexed as they change with time. <br>

#### II. Description of the project: <br>
This project estimates the regression model and its coefficients using price data for a particular stock and its corresponding index between the years of 2011 and 2023. The modules used for this project are matplotlib, yfinance, scipy, statsmodels, csv and pytest. <br> <br>
The project takes in user input for the stock ticker and the index ticker. This is validated by the program. If the input is valid, adjusted close price data for both the stock and index is retrieved using the yfinance module. This data is organized as a list and is passed into a function that calculates the stock and index daily returns(as a percentage) and subtracts the risk-free rate from it to calcuate excess index/stock returns. Whilst the original regression model assumes that risk-free rate is a constant, the project assumes that risk-free rate does change over time and data on that is also collected from a csv file and organized. Following this, a linear regression of excess stock returns on excess index returns is run and a plot of the line of best fit as well as a regression table is displayed. The regression table contains details about the coefficients, the R-value(a measure of how well the model explains the data), the t-values (statistics obtained from a hypothesis test) for each of the coefficients and more. The table generated also assumes that standard errors for the data are heteroskedastic (i.e. the variance of the error term is not constant over the range of values of excess market returns). <br> <br>
The test file for the project tests whether input validation is done properly by the program by inputting in erroneous values for tickers and checking if the program raises a value error. <br>

#### III. Limitations of the project: <br>
The limitations of this project coincide with the limitations of the CAPM model. It is likely that other variables also influence excess stock returns and not including these variables could generate an omitted variable bias whereby the estimators for both coefficients are not consistent (they do not converge in probability to the true value of beta). This means that there might be a high probability of the coefficient estimates being inaccurate, even in large samples. Moreover, another limitation could be that the excess stock returns and excess index returns might not even be linearly related, in which case it is best to use a different model to estimate these coefficients.
