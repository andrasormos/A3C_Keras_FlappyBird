import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GameEngine import PlayGame

GE = PlayGame()
games = 1
gameLength = 168
initTimerange = 24 #1460
timeStepSize = "H"

GE.startGame(gameLength, initTimerange, timeStepSize)
df_segment = GE.getChartData()


def scale_list(x, to_min, to_max):
    def scale_number(unscaled, to_min, to_max, from_min, from_max):
        return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min

    if len(set(x)) == 1:
        print("SET(X) == 1")
        return [np.floor((to_max + to_min) / 2)] * len(x)
    else:
        return [scale_number(i, to_min, to_max, min(x), max(x)) for i in x]

TIME_RANGE = initTimerange
PRICE_RANGE = initTimerange
half_scale_size = int(PRICE_RANGE / 2)

stock_closes = df_segment["Close"]
roundedCloses = ['%.2f' % elem for elem in stock_closes]
stock_closes = stock_closes[::-1]
close_data_together = list(np.round (scale_list (stock_closes[TIME_RANGE - TIME_RANGE : TIME_RANGE], 0, half_scale_size - 1), 0) )
graph_close = close_data_together[0:PRICE_RANGE]

# TOP HALF
blank_matrix_close = np.zeros(shape=(half_scale_size, TIME_RANGE))
x_ind = 0

for c in graph_close:
    blank_matrix_close[int(c), x_ind] = 1
    x_ind += 1
blank_matrix_close = blank_matrix_close[::-1]

# BOTTOM HALF
blank_matrix_diff = np.zeros(shape=(half_scale_size, TIME_RANGE))
x_ind = 0
for v in graph_close:
    blank_matrix_diff[int(v), x_ind] = 0
    x_ind += 1
blank_matrix_diff = blank_matrix_diff[::-1]

# TOP + BOTTOM
blank_matrix = np.vstack([blank_matrix_close, blank_matrix_diff])



print("Closes:", roundedCloses)
print("Graph Close:", graph_close)
print("Stock Close:", len(stock_closes))
print("Graph Close:", len(graph_close))


# PLOT
if 1 == 1:
    # graphed on matrix
    #plt.imshow(blank_matrix, cmap='hot')
    #plt.show()

    # straight timeseries
    #plt.plot(graph_close, color='black')
    #plt.show()

    fig = plt.figure()
    plt.title("Actual Closes")
    ax1 = fig.add_subplot(111)
    ax1.plot(df_segment.index, df_segment["Close"], color='b', markersize=0.6)

    ax2 = fig.add_subplot(211)
    plt.imshow(blank_matrix, cmap='hot')

    plt.show()