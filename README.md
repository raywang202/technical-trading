# Exploration of seasonal trading strategies

## Motivation

It seems plausible that markets exhibit seasonality, which may be driven by market players acting in a regular manner year after year. For example, maybe large market players make purchases of stocks are regular times of the year, driving up prices when they make purchases. Or take the phenomenon that markets in Europe tend to slow down significantly during the summer holidays, as traders go on vacation. Here I explore some simple seasonal trading strategies, built around longing/shorting stocks based off of their performance in previous years over a similar time period. I then overlay a random forest model on top of this "first stage" model, to identify strong performing stocks among those proposed by a seasonal strategy, rather than trading all stocks proposed by a seasonal strategy. I apply this method and simulate trades in 2022 and 2023, and find that the random forest seasonal strategy outperforms both the simple seasonal strategy as well as the S&P 500. And while there are a large proportion of "missed" trading opportunities due to the random forest generating a large number of false negatives (low recall), there are still enough over the course of the year that the random forest strategy comes out on top, especially if one considers trade frictions.

## Contents

1) [Initial Exploration.ipynb](Initial_Exploration.ipynb) provides evidence of seasonality in the S&P 500, and proposes a simple trading strategy that improves Sharpe/Sortino ratios compared to just holding the S&P 500. This provides a motivation for...
2) [Stock_technical.ipynb](Stock_technical.ipynb) which considers taking multiple long and short seasonal trades throughout the year, by picking the best performing stocks in the S&P 500 in terms of seasonal performance. Not surprisingly, this strategy is guilty of multiple hypothesis testing ("cherry picking") and underperforms what would have been predicted purely from historical behaviors. However, the strategy generally still has favorable average returns relative to the S&P 500, although at a significant increase in volatility and decrease in Sharpe ratio.
3) [Stock_ML_technical.ipynb](Stock_ML_technical.ipynb) refines the simple stock seasonal strategy by using a random forest classifier to identify very profitable trades (2%+ returns). The features include other seasonal trend values (historical winrates) but also near-term market condition information, such as momentum or volatility. Using this classifier on top of the seasonal model results in a large number of missed trading opportunities (low recall), but this is offset by an increase in precision, and ultimately better portfolio performance throughout 2022 and 2023 (taking fewer, more profitable trades is better than trading more often with lower average returns).

Other files are either pre-processed or downloaded data (in CSV form) or the underlying code (.py files).
