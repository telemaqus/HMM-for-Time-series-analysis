# AAPL Stock Forecasting with HMM, LSTM and ARIMA

In this project, I explored different ways to model daily Apple (AAPL) price movements. I compared a Gaussian Hidden Markov Model, an LSTM and an ARIMA model, with a focus on one question: should a model predict the closing price itself, or the relative move from the opening price?

## Predicting a relative move

A stock's dollar price changes over time, so the same dollar move means different things at different price levels. For the main experiments, I used the fractional change from Open to Close:

`(Close - Open) / Open`

I also used the day's relative High and Low as features. To convert a predicted fractional change back to a price, I used:

`predicted Close = Open × (1 + predicted fractional change)`

I included a separate Close-only LSTM as a contrast. The goal was to see how predicting the price level differs from predicting a daily move. The Close-only result should be treated cautiously; a scaling issue in that experiment affects the plotted comparison, as noted below.

## Models

- **Gaussian HMM:** I fitted a four-state model to the relative OHLC features. For each prediction, the notebook scores candidate next-day feature values from a grid and selects the candidate with the highest sequence log-likelihood.
- **LSTM:** I used the previous 10 days of relative OHLC features to predict the next day's features, then converted the predicted fractional change to a Close price.
- **ARIMA(2, 1, 2):** I examined the Close series and its first difference with ACF/PACF plots and an Augmented Dickey–Fuller test. I then forecast the intraday fractional change one step at a time, adding each observed test value to the history before the next forecast.
- **Close-only LSTM:** As a contrast, this model uses 10 scaled Close values to predict the next Close.

The data is daily AAPL OHLC history from Yahoo Finance. The train/test split is chronological, so earlier observations are used for training and later observations for testing.
