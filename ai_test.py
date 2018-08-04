import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import wrapped_flappy_bird as game
import GameEngine as game
import csv

THREADS = 1

game_state = []

for i in range(0,THREADS):
    print("init THREAD")
    game_state.append(game.PlayGame())

games = 10000

actions = [[1,0],[0,1]]

x = []
y = []

for game in range(games):
    terminal = False
    done = False

    #for step in range(gameLength):
    while terminal == False:
        a_t = actions[np.random.choice(np.arange(0, 2), p=[0.5, 0.5])]
        x_t, r_t, terminal = game_state[0].nextStep(a_t)

    if terminal == True:
        profit =  game_state[0].fullBalance -  game_state[0].initialBalance
        reward =  game_state[0].reward
        print("Game:", game, "Profit:", profit, "Reward:", reward)

        y.append(profit)
        x.append( game_state[0].startDate)
        df = pd.DataFrame({"hours": x, "profits": y})
        df.to_csv("aiTestLog.csv")

        game_state[0].startGame()


'''
df = pd.DataFrame({ "hours" : x, "profits" : y })
df["hours"] = pd.to_datetime((df["hours"]))
df = df.set_index("hours")

fig, ax = plt.subplots()
ax.plot(df.index, df["profits"], ".", markersize=1 )

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

fig.autofmt_xdate()
plt.show()
'''