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
moving_average_50 = pd.rolling_mean(stock_data['Close'], 50)
moving_average_100 = pd.rolling_mean(stock_data['Close'], 100)

# Plots all values
plt.scatter(days, stock_data['Close'], color = 'black')
plt.plot(days, model.predict(days), color = 'blue')
plt.plot(days, moving_average_50, color = 'green')
plt.plot(days, moving_average_100, color = 'red')
plt.plot(days_ahead, model.predict(days_ahead), color = 'orange')

plt.show()

# Prints suggestions for future
trending_up = model.coef_ > 0
print("Linear Regression says that: ")
if(trending_up):
    print("\tStock is going upwards - buy now!")
else:
    print("\tStock is going down - sell now!")
