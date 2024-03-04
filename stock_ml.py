import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import ssl
from itertools import product
import sklearn


# Repeat the prior data mining example to identify the top 10 long/short trades
# using the last 10 years. But apply ML to determine whether we should actually
# take those trades offered by the seasonal strategy. In this case, conditional
# on a signal, assign a 1 or 0 if the strategy would have made money, and use
# ML of this on a number of other indicators:
# 1) Other seasonal indicators: Winrate for last year, last 3 yrs, last 5 yrs
# 2) Estimate of daily volatility (EWM with span parameter = 30)
# 3) Simple momentum indicator for the stock and the SP500 overall: 'up'
#    if EWM with span of 5 (1 week) exceeds the EWM with span of 10 (2 weeks)\
# 4) Returns relative to yesterday's price, for stock and SP500 overall

def seasonal_return(data, symbol, start_date, end_date, first_year, last_year):
    data_list = []
    # Deal with Feb 29: assign start/end dates to Mar 1
    if start_date == '02-29': start_date = '03-01'
    if end_date == '02-29': end_date = '03-01'
    for year in range(first_year, (last_year + 1)):
        full_start_date = str(year) + '-' + start_date
        full_end_date = str(year) + '-' + end_date
        trade_start_date = data.loc[full_start_date, 'Price Date']
        trade_end_date = data.loc[full_end_date, 'Price Date']
        start_price = data.loc[full_start_date, symbol]

        # If price data is missing, skip that year
        if np.isnan(start_price):
            continue
        end_price = data.loc[full_end_date, symbol]
        if np.isnan(end_price):
            continue
        returns = log(end_price / start_price)
        data_list.append([symbol, year, trade_start_date, start_price,
                          trade_end_date,
                          end_price, returns])

    df = pd.DataFrame(data_list, columns=['Symbol', 'Year', 'Init Date',
                                          'Init Price', 'Final Date', 'Final Price', 'Return'])
    return df


def return_stats(x, risk_free_rate=0):
    d = {}
    d['N'] = x['Symbol'].count()
    d['avg r'] = x['Return'].mean()
    d['vol'] = x['Return'].std()
    downsides = x[x['Return'] < risk_free_rate]['Return']
    d['downside dev'] = 0 if downsides.count() == 0 else downsides.std()
    upsides = x[-x['Return'] < risk_free_rate]['Return']
    d['upside dev'] = 0 if upsides.count() == 0 else upsides.std()
    d['up'] = sum(x['Return'] > risk_free_rate)
    return pd.Series(d, index=['N', 'avg r', 'vol', 'downside dev', 'upside dev',
                               'up'])


# Identify long and short positions using the same seasonal strategy/criteria,
# using the 10-year seasonal lookback, but conducting the trades over
# 2019-2021. We will look at seasonal activity going back to 2009.

# Pull S&P500 Data

# Pull the adjusted close prices off Yahoo Finance
df = pd.read_csv("S&P500-Symbols.csv")
tickers = list(df['Symbol'])
start_date = '1989-01-01'
end_date = '2024-01-03'  # Get data a few days past end of year to backfill

# Either pull from Yahoo finance, or for read the pre-downloaded CSV
# data = pd.DataFrame(yf.download(tickers, start_date, end_date)['Adj Close'])
# data.reset_index().to_csv("S&P500-adjusted-close.csv", index=False)
data = pd.read_csv('S&P500-adjusted-close.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
all_dates = pd.date_range(start_date, end_date)
data['Price Date'] = data.index

# Backfill with trading prices for missing dates
data = data.reindex(all_dates, method='bfill')
sp500_dates_added = pd.read_csv("S&P500-Info.csv")[['Symbol', 'Date added']]

all_stocks = data.columns.drop(labels='Price Date')

# Only keep through end of 2023
data = data[data.index < '2024-01-01']

# Restrict to stocks that have data back in 2009, total of 432
sub_cols = data.columns[data.loc['2009-01-01'].notna()]
sub_cols = sub_cols.append(pd.Index(['Price Date']))
sub_stocks = data[sub_cols].columns.drop(labels='Price Date')

sub_data = data[sub_cols][data.index >= '2009-01-01']

hold_range = [7, 14, 28]  # hold for a fixed number of weeks, up to a month
delay_range = [0, 5, 10]
start_months = list(range(1, 12 + 1))
start_days = ['-01', '-15']
initial_dates = [str(i) + j for i, j in product(start_months, start_days)]

# Do this one time for each year in 2019 to 2021. Takes about 20 minutes each
all_returns_list = []

for trade_year in [2019, 2020, 2021]:
    start_year = trade_year - 10
    end_year = trade_year - 1


    for initial_date in initial_dates:

        initial_calendar_year = datetime.strptime(
            str(trade_year) + "-" + initial_date, "%Y-%m-%d")

        # Delay refers to how many days after the 1st or 15th we start the trade
        # This technically means we will never start a position on the 29-31st
        for delay in delay_range:
            start_calendar_year = initial_calendar_year + timedelta(days=delay)
            start_calendar = start_calendar_year.strftime('%m-%d')

            for hold_length in hold_range:
                end_calendar = (start_calendar_year + timedelta(days=hold_length)
                                ).strftime('%m-%d')
                stock_returns_list = []

                for stock in sub_stocks:
                    stock_returns_list.append(seasonal_return(sub_data, stock,
                                                              start_calendar, end_calendar, start_year, end_year))

                seasonal_returns = pd.concat(stock_returns_list)
                symbol_stats = seasonal_returns.groupby('Symbol').apply(
                    return_stats, risk_free_rate=0)
                symbol_stats['trade window'] = initial_calendar_year.strftime('%Y-%m-%d')
                symbol_stats['start date'] = start_calendar_year
                symbol_stats['end date'] = end_calendar
                symbol_stats['Sharpe Long'] = symbol_stats['avg r'] / symbol_stats['vol']
                symbol_stats['Sharpe Short'] = -symbol_stats['avg r'] / symbol_stats['vol']

                symbol_stats['hold length'] = hold_length
                all_returns_list.append(symbol_stats)

all_returns = pd.concat(all_returns_list)

# Annualize returns
all_returns['annualized r'] = (all_returns['avg r'] * 365 /
                               all_returns['hold length'])

# all_returns.to_csv('seasonal_trades_2019_2021.csv')


## Import the CSV of all returns

all_returns = pd.read_csv('seasonal_trades_2019_2021.csv')

all_returns['trade year']=all_returns['trade window'].str.slice(0,4)


# Consider first only the long positions
# Recall that our criteria is an annualized return > 40%, 6/10+ winrate in the last 10 years,
# and then sorting by highest Sharpe ratio using historical seasonal trades for
# each trade window

long_positions = all_returns[(all_returns['annualized r'] > 0.5) & 
(all_returns.up >= 6) & all_returns['avg r']>.01].sort_values(
    'Sharpe Long', ascending=False).groupby('trade window').head(10)

# Define the feature functions, keeping in mind that all_returns already has
# the seasonal data calculated

# Historical winrates
long_positions['past1yr']=long_positions.apply(lambda row: historical_up_rate(
    data, row['Symbol'],row['start date'][5:10], row['end date'], 
    int(row['trade year'])-1,int(row['trade year'])-1), axis = 1)

long_positions['past3yr']=long_positions.apply(lambda row: historical_up_rate(
    data, row['Symbol'],row['start date'][5:10], row['end date'], 
    int(row['trade year'])-3,int(row['trade year'])-1), axis = 1)

long_positions['past5yr']=long_positions.apply(lambda row: historical_up_rate(
    data, row['Symbol'],row['start date'][5:10], row['end date'], 
    int(row['trade year'])-5,int(row['trade year'])-1), axis = 1)

# Daily volatility with exponentially weighted moving average
# We get in a bit of trouble with the backfilled data: so move forward to
# the day we would trade, and then only consider averages of actual trading days

data_no_backfill = data[data.index==data['Price Date']]

# Note that these functions are shifted by 1, since it would involve knowing
# the current day's closing price (assume we trade DURING the day and take the
# then unknown close price)

def get_ewm_vol(data, symbol, date, span = 30):
    stock_rets = np.log(data[symbol] / data[symbol].shift(1)).shift(1)
    return stock_rets.ewm(span).std().loc[date]

long_positions['ewm_vol'] = long_positions.apply(lambda row: get_ewm_vol(
    data_no_backfill, row['Symbol'], data.loc[row['start date']]['Price Date'],
     30), axis = 1)


# EWMA of simple price: compare between a long/short window to determine
# whether the signal is long
def get_ewm_momentum(data, symbol, date, long_window, short_window):
    near_momentum_long = (data[symbol].ewm(short_window).mean().shift(1).loc[date]>
        data[symbol].ewm(long_window).mean().shift(1).loc[date])
    if near_momentum_long:
        return 1
    else:
        return 0

long_positions['long_momentum'] = long_positions.apply(lambda row:
    get_ewm_momentum(data_no_backfill, row['Symbol'], data.loc[row['start date']]['Price Date'],
        long_window = 10, short_window = 5), axis = 1)

def get_recent_return(data, symbol, date, days_back = 1):
    stock_rets = np.log(data[symbol] / data[symbol].shift(days_back)).shift(1)

long_positions['yesterday_ret'] = long_positions.apply(lambda row:
    get_recent_return(data_no_backfill, row['Symbol'],
        data.loc[row['start date']]['Price Date'],days_back = 1), axis = 1)



# Get the actual returns for the start/end date

def get_actual_return(data, symbol, start_date, end_date):
    return np.log(data[symbol].loc[end_date]/data[symbol].loc[start_date])

long_positions['actual_return'] = long_positions.apply(lambda row:
    get_actual_return(data, row['Symbol'], row['start date'],
        row['trade year']+'-'+row['end date']), axis = 1)

# Assign 1 or 0 for whether or not we should trade: this is the outcome variable
# to compare vs our features
# Here, due to the presence of slippage, let's restrict to at least a 1%
# absolute return. Even in that case, 351/720 long trades would have absolute
# returns over 1%
long_positions['outcome'] = np.where(
    long_positions['actual_return'] > 0.01, 1, 0)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report

from sklearn.model_selection import RandomizedSearchCV


X = long_positions[['past1yr','past3yr','past5yr','ewm_vol','long_momentum',
'yesterday_ret']].values
y = long_positions['outcome'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8,
                                                    shuffle=False)

# Search amongst some common hyperparameters
param_grid = {'max_depth': [3, 5, 7],
              'max_features':np.arange(0.1, 1, 0.1),
              'max_samples': [0.3, 0.5, 0.8],
              'n_estimators':np.arange(50,500,50)}

model = RandomizedSearchCV(RandomForestClassifier(),
    param_grid, n_iter = 40, cv = 4).fit(X_train, y_train)
model.best_params_

model.fit(X_train,y_train)

y_pred_rf = model.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred, 
                            target_names=['no_trade','trade']))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

print(classification_report(y_test,np.repeat(1,len(y_test)),
                            target_names=['no_trade_no_ML','trade_no_ML']))

# Now see what would happen if we did this in 2022/2023?

later_returns_list = []

for trade_year in [2022, 2023]:
    start_year = trade_year - 10
    end_year = trade_year - 1


    for initial_date in initial_dates:

        initial_calendar_year = datetime.strptime(
            str(trade_year) + "-" + initial_date, "%Y-%m-%d")

        # Delay refers to how many days after the 1st or 15th we start the trade
        # This technically means we will never start a position on the 29-31st
        for delay in delay_range:
            start_calendar_year = initial_calendar_year + timedelta(days=delay)
            start_calendar = start_calendar_year.strftime('%m-%d')

            for hold_length in hold_range:
                end_calendar = (start_calendar_year + timedelta(days=hold_length)
                                ).strftime('%m-%d')
                stock_returns_list = []

                for stock in sub_stocks:
                    stock_returns_list.append(seasonal_return(sub_data, stock,
                                                              start_calendar, end_calendar, start_year, end_year))

                seasonal_returns = pd.concat(stock_returns_list)
                symbol_stats = seasonal_returns.groupby('Symbol').apply(
                    return_stats, risk_free_rate=0)
                symbol_stats['trade window'] = initial_calendar_year.strftime('%Y-%m-%d')
                symbol_stats['start date'] = start_calendar_year
                symbol_stats['end date'] = end_calendar
                symbol_stats['Sharpe Long'] = symbol_stats['avg r'] / symbol_stats['vol']
                symbol_stats['Sharpe Short'] = -symbol_stats['avg r'] / symbol_stats['vol']

                symbol_stats['hold length'] = hold_length
                all_returns_list.append(symbol_stats)

later_returns = pd.concat(all_returns_list)

# Annualize returns
later_returns['annualized r'] = (all_returns['avg r'] * 365 /
                               all_returns['hold length'])

# later_returns.to_csv('seasonal_trades_2022_2023.csv')
later_returns = pd.read_csv('seasonal_trades_2022_2023.csv')


later_long_positions = later_returns[(later_returns['annualized r'] > 0.4) & 
(later_returns.up >= 6)].sort_values('Sharpe Long', ascending=False).groupby(
    'trade window').head(10)

# Define the feature functions, keeping in mind that all_returns already has
# the seasonal data calculated

# Historical winrates
later_long_positions['past1yr']=later_long_positions.apply(lambda row: historical_up_rate(
    data, row['Symbol'],row['start date'][5:10], row['end date'], 
    int(row['trade year'])-1,int(row['trade year'])-1), axis = 1)

later_long_positions['past3yr']=later_long_positions.apply(lambda row: historical_up_rate(
    data, row['Symbol'],row['start date'][5:10], row['end date'], 
    int(row['trade year'])-3,int(row['trade year'])-1), axis = 1)

later_long_positions['past5yr']=later_long_positions.apply(lambda row: historical_up_rate(
    data, row['Symbol'],row['start date'][5:10], row['end date'], 
    int(row['trade year'])-5,int(row['trade year'])-1), axis = 1)

later_long_positions['ewm_vol'] = later_long_positions.apply(lambda row: get_ewm_vol(
    data, row['Symbol'], row['start date'], 30), axis = 1)

later_long_positions['long_momentum'] = later_long_positions.apply(lambda row:
    get_ewm_momentum(data, row['Symbol'], row['start date'],
        long_window = 10, short_window = 5), axis = 1)

long_2022 = later_long_positions[later_long_positions['trade year']==2022]
X_2022 = long_2022[['past1yr','past3yr','past5yr','ewm_vol','long_momentum']].values
y_2022 = np.where(long_2022['actual_return'] > 0.01, 1, 0)

long_2023 = later_long_positions[later_long_positions['trade year']==2023]
X_2023 = long_2023[['past1yr','past3yr','past5yr','ewm_vol','long_momentum']].values
y_2023 = np.where(long_2023['actual_return'] > 0.01, 1, 0)