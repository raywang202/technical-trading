import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import ssl

# Part 1

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
tickers = list(df['Symbol'])
start_date = '1989-01-01'
end_date = '2024-01-03'  # Get data a few days past end of year to backfill

# Pull off Yahoo finance the adjusted close prices
# data = pd.DataFrame(yf.download(tickers, start_date, end_date)['Adj Close'])
# data.reset_index().to_csv("S&P500-adjusted-close.csv", index=False)
data = pd.read_csv('S&P500-adjusted-close.csv')
data['Date']= pd.to_datetime(data['Date'])
data = data.set_index('Date')
all_dates = pd.date_range(start_date, end_date)
data['Price Date'] = data.index

# Backfill with trading prices for missing dates
data = data.reindex(all_dates, method='bfill')
sp500_dates_added = pd.read_csv("S&P500-Info.csv")[['Symbol','Date added']]

all_stocks = data.columns.drop(labels='Price Date')

# Only keep through end of 2023
data = data[data.index < '2024-01-01']


# Part 2

# Define a function that outputs historical performance between two dates
# Note that this specific function does not work for start and end dates that
# cross a year, e.g. a Dec-Jan seasonal trend

def seasonal_return(data, symbol, start_date, end_date, first_year, last_year):
    data_list = []
    for year in range(first_year, (last_year+1)):
        full_start_date = str(year)+'-'+start_date
        full_end_date = str(year)+'-'+end_date
        trade_start_date = data.loc[full_start_date,'Price Date']
        trade_end_date = data.loc[full_end_date,'Price Date']
        start_price = data.loc[full_start_date,symbol]

        # If price data is missing, skip that year
        if np.isnan(start_price):
            continue
        end_price = data.loc[full_end_date,symbol]
        if np.isnan(end_price):
            continue
        returns = (end_price/start_price)-1
        data_list.append([symbol, year, trade_start_date, start_price, 
            trade_end_date,
            end_price, returns])

    df = pd.DataFrame(data_list, columns=['Symbol','Year','Init Date',
        'Init Price','Final Date','Final Price','Return'])
    return df

stock_returns_list = []
start_calendar = '01-28'
end_calendar = '02-02'
start_year = 2014
end_year = 2023
for stock in all_stocks:
    stock_returns_list.append(seasonal_return(data, stock, start_calendar, 
    end_calendar, start_year, end_year))

seasonal_returns = pd.concat(stock_returns_list)

def sortino(df, strat_name, risk_free, threshold):
    excess_return = df[strat_name]-df[risk_free]
    downside = excess_return[(excess_return<df[threshold])]
    denom = np.sqrt(sum(downside*downside)/len(downside))
    return excess_return.mean()/denom

# Generate return stats
def return_stats(x, risk_free_rate = 0):
    d = {}
    d['N'] = x['Symbol'].count()
    d['avg r'] = x['Return'].mean()
    d['vol'] = x['Return'].std()
    downsides = x[x['Return'] < risk_free_rate]['Return']
    d['downside dev'] = 0 if downsides.count()==0 else downsides.std()
    upsides = x[x['Return'] > risk_free_rate]['Return']
    d['upside dev'] = 0 if upsides.count()==0 else upsides.std()
    d['up'] = sum(x['Return']>risk_free_rate)
    return pd.Series(d, index = ['N','avg r','vol','downside dev','upside dev',
        'up'])

# N is number of observations, avg r is average return for going LONG,
# vol is std dev of returns, downside/upside dev are corresponding deviations
# used to calculate Sortino Ratio, and up is number of observations that are
# above the risk free rate
symbol_stats = seasonal_returns.groupby('Symbol').apply(return_stats, risk_free_rate = 0)

symbol_stats['Sharpe Long'] = symbol_stats['avg r']/symbol_stats['vol']
symbol_stats['Sharpe Short'] = -symbol_stats['avg r']/symbol_stats['vol']
symbol_stats['Sortino Long'] = symbol_stats['avg r']/symbol_stats['downside dev']
symbol_stats['Sortino Short'] = -symbol_stats['avg r']/symbol_stats['upside dev']
symbol_stats['Winrate Long'] = symbol_stats['up']/symbol_stats['N']


# Identify the five most profitable seasonal trades for long/short
# Restrict to samples that we actually have 10 years of full data for
sub_stats = symbol_stats[symbol_stats.N==10]
best_long_trades = sub_stats.sort_values(by='avg r',ascending= False).iloc[0:5]
best_short_trades = sub_stats.sort_values(by='avg r',ascending= True).iloc[0:5]


# Part 3

# Data mine amongst all stocks, for a range of holding periods

# Hold between 5 to 30 days
hold_range = range(5,7+1,1)
start_calendar = '01-29'
start_year = 2014
end_year = 2023

all_returns_list = []

for hold_length in hold_range:
    # Use current year of 2024: this only affects whether we consider Feb 29
    # for holding windows in outputting results, but is irrelevant when looking
    # at historicals
    start_calendar_2024 = datetime.strptime("2024-"+start_calendar, "%Y-%m-%d")
    end_calendar = (start_calendar_2024+timedelta(days=hold_length)
        ).strftime('%m-%d')

    stock_returns_list = []
    for stock in all_stocks:
        stock_returns_list.append(seasonal_return(data, stock, start_calendar, 
        end_calendar, start_year, end_year))

    seasonal_returns = pd.concat(stock_returns_list)
    symbol_stats = seasonal_returns.groupby('Symbol').apply(
        return_stats, risk_free_rate = 0)

    symbol_stats['Sharpe Long'] = symbol_stats['avg r']/symbol_stats['vol']
    symbol_stats['Sharpe Short'] = -symbol_stats['avg r']/symbol_stats['vol']
    symbol_stats['Sortino Long'] = symbol_stats['avg r']/symbol_stats['downside dev']
    symbol_stats['Sortino Short'] = -symbol_stats['avg r']/symbol_stats['upside dev']

    symbol_stats['hold length'] = hold_length
    all_returns_list.append(symbol_stats)

all_returns = pd.concat(all_returns_list)

# Approximately annualize the returns (365 days)
all_returns['annl r'] = all_returns['avg r']*365/all_returns['hold length']