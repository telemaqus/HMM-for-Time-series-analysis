# AAPL Time-Series Forecasting: HMM, LSTM, and ARIMA

An exploratory financial time-series project comparing three approaches to forecasting Apple (AAPL) daily prices: a Gaussian Hidden Markov Model, an LSTM, and an ARIMA baseline. The notebook also investigates an important target-design choice: predicting a scale-relative daily move rather than the absolute closing-price level.

## The question behind the project

A model trained directly on a stock's Close price can learn the broad price level or trend and produce a curve that appears to follow the data, while still failing to capture the day-to-day movement of interest. This project therefore represents the daily move from Open to Close as a fraction of Open:

\[
\text{intraday return} = \frac{\text{Close} - \text{Open}}{\text{Open}}.
\]

The same feature set also includes the relative intraday High and Low. A predicted fractional move is converted back into a price using the known Open:

\[
\widehat{\text{Close}} = \text{Open} \times (1 + \widehat{\text{intraday return}}).
\]

This makes the prediction target less dependent on the absolute dollar scale of AAPL and focuses the models on relative movement. A separate Close-only LSTM experiment provides a contrast and motivates this choice.

## Approaches

- **Gaussian HMM:** fits a four-state model to the relative OHLC features. For each forecast, it searches a finite grid of candidate next-day feature vectors and selects the one with the highest sequence log-likelihood. The selected relative change is converted to a Close price.
- **Feature-based LSTM:** uses the previous 10 days of the same three relative features to predict the next feature vector. Its predicted Close is reconstructed from the predicted fractional move and the day's Open.
- **ARIMA(2, 1, 2):** first examines the Close series and its first difference with ACF/PACF plots and an Augmented Dickey–Fuller test. It then forecasts the intraday fractional change in a walk-forward loop, incorporating each observed test-period change before forecasting the next day.
- **Close-only LSTM contrast:** uses 10 scaled Close values to predict the next Close. It demonstrates why the target representation matters: the absolute level can encourage a model to track the broad series level instead of the local movement.

The HMM and LSTM use chronological train/test splits; ARIMA also trains on the earlier portion and forecasts into the later portion. The data source is daily AAPL OHLC history downloaded from Yahoo Finance.

## Reading the results

The notebook contains prediction plots that place actual and predicted Close values on the same chart. It also reports an ADF p-value of approximately `7.97 × 10⁻²⁸` for the first-differenced Close series, which is strong evidence against a unit root in that differenced series under the test assumptions. This supports the notebook's exploration of changes rather than treating the raw price level as stationary.

The plots are exploratory, not a model leaderboard. HMM predictions are shown for test positions 200–299, the feature LSTM for positions 400–499, and ARIMA for the first 100 test observations. Since the models are not evaluated on a common window and the notebook does not calculate shared error or directional metrics, the figures do not establish that one method is more accurate than another.

## Important limitations

- The Close-only LSTM is a diagnostic contrast, not a definitive test that absolute-price forecasting cannot work. In its current implementation, the plotted `actual` values are rescaled without adding back the minimum Close, while predictions do add it. The min/max values are also calculated before the train/test split, allowing test-period information into preprocessing. These issues can distort that comparison.
- The HMM chooses the best candidate from a predefined grid. Its selected candidate is not a calibrated probability forecast and is limited by the grid resolution.
- The prediction windows differ between models, and no common MAE/RMSE or directional-accuracy table is provided. Model ranking would require a shared evaluation window and metrics.
- The analysis is not a trading strategy: it does not account for transaction costs, slippage, or risk, and it makes no claim of investment performance.
- Yahoo Finance is an external data source, so rerunning the notebook requires network access and may retrieve revised or differently formatted data.

## Notebook

Open [`Hidden_Markov_Model.ipynb`](Hidden_Markov_Model.ipynb) for the full analysis. Explanatory Markdown sections have been added around the existing work; the original code cells, outputs, and headings are preserved. The original file in Downloads is untouched.

The notebook uses Python with NumPy, pandas, yfinance, hmmlearn, PyTorch, scikit-learn, Matplotlib, statsmodels, and tqdm. Re-running it requires those packages and access to Yahoo Finance. Since the code has intentionally been left unchanged, its existing execution-order/setup issues have not been repaired in this presentation copy; the saved outputs are preserved as the author's original results.
