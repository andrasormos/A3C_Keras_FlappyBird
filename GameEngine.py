from random import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PlayGame(object):
    def __init__(self):
        self.gameIsRunning = True

        # LOAD DATA
        dateParse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
        self.df = pd.read_csv("Gdax_BTCUSD_1h_close.csv", parse_dates=["Date"], date_parser=dateParse, index_col=0)
        self.startGame()
        self.gameStep = 0

    def startGame(self):
        self.gameLength = 84  # How long the game should go on
        self.initialTimeRange = 84  # How many data increment should be shown as history. Could be hours, months
        self.timeStepSize = "H"  # Does nothing atm
        self.amountToSpend = 500  # How much to purchase crypto for
        self.initialBalance = 50000 # Starting Money

        self.cashBalance = self.initialBalance
        self.BTC_Balance = 0  # BTC to start with
        self.actionTaken = 0

        if self.timeStepSize == "D":
            self.df = self.df.resample("D").mean()

        self.dataSize = len(self.df.index)

        # GET RANDOM SEGMENT FROM DATA
        self.startDate, self.endDate, self.startIndex, self.endIndex = self.randomChart()
        self.df_segment = self.df.loc[self.startDate: self.endDate]

        self.currentBTCPrice = 0

        self.fullBalance = self.cashBalance
        self.prevFullBalance = self.fullBalance
        self.getInitBTCPrice()
        self.rekt = False
        self.done = False
        self.cnt = 1
        self.reward = 0
        self.profit = 0
        self.realProfit = 0

    def getInitBTCPrice(self):
        endIndex = self.endIndex - 1
        endDate = self.df.index[endIndex]
        nextRow = self.df.loc[[endDate]]
        self.currentBTCPrice = nextRow["Close"][0]

    def randomChart(self):
        if self.timeStepSize == "H":
            startIndex = randint((self.initialTimeRange + self.gameLength),
                                 self.dataSize)  # self.initialTimeRange # self.dataSize
            endIndex = startIndex - self.initialTimeRange + 1

        if self.timeStepSize == "D":
            startIndex = randint(0, (self.dataSize - self.initialTimeRange + 1))
            endIndex = startIndex + self.initialTimeRange

        print("Random Chart:", startIndex, " - ", endIndex)
        startDate = self.df.index[startIndex]
        endDate = self.df.index[endIndex]

        if self.timeStepSize == "H":
            startDateStr = startDate.strftime("%Y-%m-%d %H:%M:%S")
            endDateStr = endDate.strftime("%Y-%m-%d %H:%M:%S")

        if self.timeStepSize == "D":
            startDateStr = startDate.strftime("%Y-%m-%d")
            endDateStr = endDate.strftime("%Y-%m-%d")


        return startDateStr, endDateStr, startIndex, endIndex

    def nextStep(self, action):
        self.gameStep += 1
        self.cnt = self.cnt + 1
        self.endIndex = self.endIndex - 1
        self.endDate = self.df.index[self.endIndex - 1]
        self.nextRow = self.df.loc[[self.endDate]]
        self.df_segment = pd.concat([self.nextRow, self.df_segment])
        self.df_segment = self.df_segment.drop(self.df_segment.index[len(self.df_segment)-1])
        self.currentBTCPrice = self.nextRow["Close"][0]

        if action == [1, 0]:
            print("-BOUGHT BTC-")
            self.actionTaken = 1
            if self.amountToSpend > self.cashBalance:
                self.cashBalance = 0
                self.BTC_Balance = round((self.BTC_Balance + (self.cashBalance / self.currentBTCPrice)), 5)
            else:
                self.cashBalance = self.cashBalance - self.amountToSpend
                self.BTC_Balance = round((self.BTC_Balance + (self.amountToSpend / self.currentBTCPrice)), 5)

        if action == [99, 99]:
            print("-SOLD BTC-")
            moneyWorthInBTC = self.amountToSpend / self.currentBTCPrice  # 0.1

            if moneyWorthInBTC > self.BTC_Balance:
                self.cashBalance = self.cashBalance + (self.BTC_Balance * self.currentBTCPrice)
                self.BTC_Balance = 0
            else:
                self.BTC_Balance = self.BTC_Balance - moneyWorthInBTC
                self.cashBalance = self.cashBalance + self.amountToSpend

        if action == [0, 1]:
            self.actionTaken = 0
            print("-SKIP-")

        self.cashBalance = round((self.cashBalance), 0)
        self.BTC_Balance = round((self.BTC_Balance), 5)
        self.fullBalance = round((self.cashBalance + (self.BTC_Balance * self.currentBTCPrice)), 0)

        self.profit = self.fullBalance - self.initialBalance
        self.realProfit = self.profit
        # REWARDING SYSTEM
        if self.profit > 0:
            self.reward = 0.05
            #print("in profit")

        if self.profit > 200:
            self.reward = 1.5
            self.resetWallet()
            #print("winning")

        if self.profit < -200:
            self.reward = -1
            self.resetWallet()
            #print("rekt")

        if self.cnt == self.gameLength:
            self.done = True

        if self.done == True:
            self.gameStep = 0

        print(self.gameStep,self.endDate, "- PROFIT:", self.profit, "BAL:", self.fullBalance, "BTC", self.BTC_Balance," CASH:", self.cashBalance,"DONE:", self.done)
        print("\n")

        image = self.getChartImage(self.initialTimeRange, self.df_segment)

        return image, self.reward, self.done

    def resetWallet(self):
        self.profit = 0
        self.BTC_Balance = 0
        self.fullBalance = 50000
        self.cashBalance = 50000

    def getActionTaken(self):
        return self.actionTaken

    def getProfit(self):
        return (self.fullBalance - self.initialBalance)

    def getChartData(self):
        image = self.getChartImage(self.initialTimeRange, self.df_segment)

        return image


# ---------------------------------     CHART IMAGE GENERATION      ---------------------------------

    def getChartImage(self, initTimerange, df_segment):

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

        #print(df_segment)
        #print("\n")

        return blank_matrix




actions = [[0,1],[1,0]]


if __name__ == "__main__":
    test = PlayGame()

    chart, r_t, terminal = test.nextStep([0, 1])
    # PLOT
    plt.imshow(chart, cmap='hot')
    plt.show()

    a_t = actions[np.random.choice(np.arange(0, 2), p=[0.5, 0.5])]

    chart, r_t, terminal = test.nextStep(a_t)
    # PLOT
    plt.imshow(chart, cmap='hot')
    plt.show()




