import pandas as pd
import yfinance as yf

# Read in the SP500 tickers
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]

# tickers = df['Symbol']
# Save to CSV if interested
# df.to_csv('S&P500-Info.csv')
# df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])


# Fetch the data
data = yf.download(tickers,'2015-1-1')['Adj Close']

# Print first 5 rows of the data
print(data.head())


