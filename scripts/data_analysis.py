import pandas as pd

def calc_moving_avrg(data, window):
    return data.rolling(window=window).mean()