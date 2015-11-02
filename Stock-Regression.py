import Quandl
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as linear_model

stock_data = Quandl.get("GOOG/NYSE_PEP", trim_start = "2014-01-01", trim_end = "2014-12-31", authtoken = "fcG4eM3axadY2vy5xkHr")

days = (stock_data.index - stock_data.index[0]).days.reshape(-1, 1)

# Linear Regression
last_day = days[len(days) - 1]
days_ahead = [[i] for i in range(last_day + 1, last_day + 30)]

model = linear_model.LinearRegression()
model.fit(days, stock_data['Close'])

# Moving Averages
moving_average_26 = pd.ewma(stock_data['Close'], span = 26)
moving_average_12 = pd.ewma(stock_data['Close'], span = 12)
MACD = moving_average_12 - moving_average_26
signal_line = pd.ewma(MACD, span = 9)

# Plots main graph
plt.scatter(days, stock_data['Close'], color = 'black')
plt.plot(days, model.predict(days), color = 'blue')
plt.plot(days, moving_average_12, color = 'green')
plt.plot(days, moving_average_26, color = 'red')
plt.plot(days_ahead, model.predict(days_ahead), color = 'orange')

plt.show()

# plot MACD in separate graph
plt.plot(days, MACD)
plt.plot(days, signal_line)

plt.show()

# Prints suggestions for future
trending_up = model.coef_ > 0
print("Linear Regression says that: ")
if(trending_up):
    print("\tStock is going upwards - buy now!")
else:
    print("\tStock is going down - sell now!")
