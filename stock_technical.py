import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import seaborn as sns

# A data-mining exercise

# Read in the SP500 tickers
# ssl._create_default_https_context = ssl._create_unverified_context
# table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# df = table[0]
# df.to_csv('S&P500-Info.csv', index=False)
# df.to_csv("S&P500-Symbols.csv", columns=['Symbol'], index=False)

# Alternatively, read tickers from pre-existing csv
df = pd.read_csv("S&P500-Symbols.csv")
tickers = df['Symbol']

# Fetch data
data = yf.download(tickers,'2015-1-1')['Adj Close']

