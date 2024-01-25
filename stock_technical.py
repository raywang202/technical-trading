import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import ssl

# A data-mining exercise
# Let's see what happens when we just trade repeatedly off of technical
# trend indicators. On a given day, using data from the past X years, identify
# the top 5 historical trades in a given index, based off of some criteria such
# as Sharpe.

# Pre-process data with forward filled fields, in this case using the SP500
# Read in the SP500 tickers from S&P500-Symbols.csv, which was pulled from
# Wikipedia and saved locally.

# Run the below code to pull symbols
# ssl._create_default_https_context = ssl._create_unverified_context
# table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# df = table[0]
# df.to_csv("S&P500-Symbols.csv", columns=['Symbol'], index=False)

# Pull the adjusted close prices off Yahoo Finance
df = pd.read_csv("S&P500-Symbols.csv")
tickers = df['Symbol']
start_date = '1989-01-01'
end_date = '2024-01-03'  # Get data a few days past end of year to backfill
# Pull off Yahoo finance the adjusted close prices
data = pd.DataFrame(yf.download(tickers, start_date, end_date)['Adj Close'])

all_dates = pd.date_range(start_date, end_date)
data['Price Date'] = data.index

# Fill in missing prices with the first price after the date
# This would be "forward-looking" if we were trading off of prices, but
# we are trading off of specific dates so we do not see into the future
data = data.reindex(all_dates, method='bfill')

# Drop data for 2024
data = data[spx.index < '2024-01-01']

# Normalize prices relative to first of the year
spx['Year'] = spx.index.year
spx['Month Day'] = spx.index.strftime('%m-%d')
spx['Yr Init Price'] = spx.groupby('Year', sort=False).transform(
    'first')['Adj Close']
spx['Seasonal Trend Price'] = spx['Adj Close'] / spx['Yr Init Price'] * 100

# Define a function that outputs historical performance between two dates
#
