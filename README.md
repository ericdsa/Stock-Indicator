# Stock-Indicator

This is a tool to analyze a stock through various statistical and quantitative methods for predicting future price movements.

The tool takes in a stock, pulls recent stock data from Quandl in the form of a pandas time series, and performs a variety of actions:
1) Creates a linear regression model
2) Creates a price channel based on the 20 day high and 20 day low for each individual point
3) Calculates 50 day and 200 day weighted moving averages
4) Calculates MACD (Moving Average Convergence Divergence) and signal line

It displays all data and analysis in a figure with matplotlib, and then provides a recommendation on whether to buy or sell the stock. It does this by taking each of the indicators and analyzing each indicator's suggested price movement (up or down). Then, it compiles an accumulated list, and the majority wins (up = buy, down = sell).
