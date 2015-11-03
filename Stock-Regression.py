import Quandl
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as linear_model

def main():
    # Gathers initial data
    stock_data, days = gather_data()

    # Linear Regression
    days_ahead, linear_reg = make_linear_reg(stock_data, days, 30)

    # Exponentially Weighted Moving Averages and MACD
    moving_average_26, moving_average_12, MACD, signal_line = make_MACD(stock_data)

    # Plots main graph of stock movement, linear regression, and EWMAs
    plt.scatter(days, stock_data['Close'], color = 'black')
    plt.plot(days, linear_reg.predict(days), color = 'blue')
    plt.plot(days, moving_average_12, color = 'green')
    plt.plot(days, moving_average_26, color = 'red')
    plt.plot(days_ahead, linear_reg.predict(days_ahead), color = 'orange')

    plt.show()

    # Plots MACD in separate graph
    plt.plot(days, MACD)
    plt.plot(days, signal_line)

    plt.show()

    # Prints suggestions for future
    trending_short = signal_line[len(signal_line) - 1] > MACD[len(MACD) - 1]
    trending_long = linear_reg.coef_ > 0
    print_suggestions(trending_short, trending_long)

def gather_data():
    stock_data = Quandl.get("GOOG/NYSE_PEP", trim_start = "2014-01-01", trim_end = "2014-12-31", authtoken = "fcG4eM3axadY2vy5xkHr")
    days = (stock_data.index - stock_data.index[0]).days.reshape(-1, 1)
    return stock_data, days

def make_linear_reg(stock_data, days, number_days_ahead):
    last_day = days[len(days) - 1]
    days_ahead = [[i] for i in range(last_day + 1, last_day + number_days_ahead + 1)]
    model = linear_model.LinearRegression()
    model.fit(days, stock_data['Close'])

    return days_ahead, model

def make_MACD(stock_data):
    moving_average_26 = pd.ewma(stock_data['Close'], span = 26)
    moving_average_12 = pd.ewma(stock_data['Close'], span = 12)
    MACD = moving_average_12 - moving_average_26
    signal_line = pd.ewma(MACD, span = 9)

    return moving_average_26, moving_average_12, MACD, signal_line

def print_suggestions(trending_short, trending_long):
    print("Results: \n")
    print("Short-Term Analysis (MACD): ")
    if(trending_short):
        print("\tBuy Now!")
    else:
        print("Sell Now!")

    print("Long-Term Analysis (Linear Regression): ")
    if(trending_long):
        print("\tBuy Now!")
    else:
        print("Sell Now!")

if __name__ == "__main__":
    main()
