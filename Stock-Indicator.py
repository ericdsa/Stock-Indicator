import Quandl
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as linear_model

def main():
    # get symbol to search for
    symbol = input("Input a stock symbol: ")

    # Gathers initial data
    stock_data, days = gather_data(symbol)

    # price channel
    back_reference_days = 20
    low_channel, high_channel = price_channel(stock_data, back_reference_days)

    # 50 and 200 day moving averages
    moving_average_50, moving_average_200 = moving_averages(stock_data)

    # Exponentially Weighted Moving Averages and MACD
    MACD, signal_line = make_MACD(stock_data)

    # Linear Regression
    days_ahead, linear_reg = make_linear_reg(stock_data, days, 30)

    # Plots main graph of stock movement, linear regression, and EWMAs
    plt.scatter(days, stock_data['Close'], color = 'black')
    plt.plot(days, linear_reg.predict(days), color = 'blue')
    plt.plot(days, low_channel, color = 'purple')
    plt.plot(days, high_channel, color = 'purple')
    plt.plot(days, moving_average_50, color = 'green')
    plt.plot(days, moving_average_200, color = 'red')
    plt.plot(days_ahead, linear_reg.predict(days_ahead), color = 'orange')

    plt.show()

    # collect indicators - True if trending up, False if trending down
    moving_averages_up = moving_average_50[len(moving_average_50) - 1] > moving_average_200[len(moving_average_200) - 1]
    MACD_up = MACD[len(MACD) - 1] > signal_line[len(signal_line) - 1]
    linear_up = linear_reg.coef_ > 0

    indicator_list = [moving_averages_up, MACD_up, linear_up]
    print(indicator_list)
    print_suggestions(indicator_list)

def gather_data(symbol):
    symbol_tag = "WIKI/" + symbol
    stock_data = Quandl.get(symbol_tag, trim_start = "2014-01-01", trim_end = "2014-12-31", authtoken = "")
    days = (stock_data.index - stock_data.index[0]).days.reshape(-1, 1)
    return stock_data, days

def price_channel(stock_data, back_reference_days):
    low_channel = pd.rolling_min(stock_data['Close'], window = back_reference_days)
    high_channel = pd.rolling_max(stock_data['Close'], window = back_reference_days)
    return low_channel, high_channel

def moving_averages(stock_data):
    moving_average_50 = pd.rolling_mean(stock_data['Close'], window = 50)
    moving_average_200 = pd.rolling_mean(stock_data['Close'], window = 200)
    return moving_average_50, moving_average_200

def make_MACD(stock_data):
    moving_average_26 = pd.ewma(stock_data['Close'], span = 26)
    moving_average_12 = pd.ewma(stock_data['Close'], span = 12)
    MACD = moving_average_12 - moving_average_26
    signal_line = pd.ewma(MACD, span = 9)

    return MACD, signal_line

def make_linear_reg(stock_data, days, number_days_ahead):
    last_day = days[len(days) - 1]
    days_ahead = [[i] for i in range(last_day + 1, last_day + number_days_ahead + 1)]
    model = linear_model.LinearRegression()
    model.fit(days, stock_data['Close'])

    return days_ahead, model

def print_suggestions(indicator_list):
    # if count of Trues is greater than count of Falses, then buy, otherwise sell
    true_count = indicator_list.count(True)
    false_count = indicator_list.count(False)
    print("Number of indicators recommending to buy: " + str(true_count))
    print("Number of indicators recommending to sell: " + str(false_count))
    if(true_count > false_count):
        print("Indicators say you should buy!")
    else:
        print("Indicators say you should sell!")

if __name__ == "__main__":
    main()
