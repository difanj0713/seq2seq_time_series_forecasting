import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_predict
import matplotlib.dates as mdates
import datetime as dt

def naive_arima(array):
    # find_differ_order(array)
    df = pd.DataFrame(array, columns=['value'])
    model = ARIMA(df.value, order=(1, 1, 2))
    model_fit = model.fit()
    print(model_fit.summary())

    time_span = array.shape[0]
    start = dt.date(2019, 9, 23)
    then = start + dt.timedelta(days=(time_span))
    days = mdates.drange(start, then, dt.timedelta(days=1))

    label = "Predicted90D"
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.plot(array, label="Actual90d")
    model_fit.predict().plot(label=label)
    plt.savefig("ArimaPrediction90D.png")