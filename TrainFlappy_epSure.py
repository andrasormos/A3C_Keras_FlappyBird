import numpy as np
import sys

sys.path.append("game/")
import skimage
from skimage import transform, color, exposure
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Activation, Input
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K
from keras.callbacks import LearningRateScheduler, History
import tensorflow as tf
import pygame
import wrapped_flappy_bird as game
import scipy.misc
import scipy.stats as st
import threading
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import random

GAMMA = 0.99  # discount value
BETA = 0.01  # regularisation coefficient
IMAGE_ROWS = 40
IMAGE_COLS = 40
IMAGE_CHANNELS = 4
LEARNING_RATE = 7e-4
EPISODE = 0
THREADS = 8
t_max = 5
const = 1e-5
T = 0

episode_r = []
episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
episode_output = []
episode_critic = []

ACTIONS = 2
a_t = np.zeros(ACTIONS)

policyLoss = 0
criticLoss = 0
lrate = 0

# EPSILON
FINAL_EPSILON = 0.01  # final value of epsilon
INITIAL_EPSILON = 0.9  # starting value of epsilon
epsilon = INITIAL_EPSILON
EXPLORE = 50000.

# Loss function for policy output
def logloss(y_true, y_pred):  # policy loss
    return -K.sum(K.log(y_true * y_pred + (1 - y_true) * (1 - y_pred) + const), axis=-1)
    # BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

# Loss function for critic output
def sumofsquares(y_true, y_pred):  # critic loss
    return K.sum(K.square(y_pred - y_true), axis=-1)

# Function buildmodel() to define the structure of the neural network in use
def buildmodel():
    print("Model building begins")

    model = Sequential()
    keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

    S = Input(shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS,), name='Input')
    h0 = Convolution2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', kernel_initializer='random_uniform',
                       bias_initializer='random_uniform')(S)
    h1 = Convolution2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='random_uniform',
                       bias_initializer='random_uniform')(h0)
    h2 = Flatten()(h1)
    h3 = Dense(256, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(h2)
    P = Dense(1, name='o_P', activation='sigmoid', kernel_initializer='random_uniform',
              bias_initializer='random_uniform')(h3)
    V = Dense(1, name='o_V', kernel_initializer='random_uniform', bias_initializer='random_uniform')(h3)

    model = Model(inputs=S, outputs=[P, V])
    rms = RMSprop(lr=LEARNING_RATE, rho=0.99, epsilon=0.1)
    model.compile(loss={'o_P': logloss, 'o_V': sumofsquares}, loss_weights={'o_P': 1., 'o_V': 0.5}, optimizer=rms)
    return model

#Function to preprocess an image before giving as input to the neural network
def preprocess(image):
    image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image, (IMAGE_ROWS, IMAGE_COLS), mode = 'constant')
    image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
    image = exposure.rescale_intensity(image, in_range=(1, 2))
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
    image = image.reshape(1, image.shape[0], image.shape[1], 1)
    return image

# Initialize a new model using buildmodel() or use load_model to resume training an already trained model
model = buildmodel()
# Model = load_model("saved_models/model_updates3900", custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
model._make_predict_function()
graph = tf.get_default_graph()

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('o_P').output)

a_t[0] = 1  # Index 0 = no flap, 1 = flap
# Output of network represents probability of flap

game_state = []
for i in range(0, THREADS):
    game_state.append(game.GameState(30000))

playLog = pd.DataFrame(columns=['score', 'random', 'predictedChoice', 'verySureMean'])
logCnt = 0
score = 0
randList = []
randMean = 0
verySureList = []
verySureMean = 0
myCount = 0

def runprocess(thread_id, s_t):
    global T
    global a_t
    global model
    global score
    global logCnt
    global trainingLog
    global epsilon
    global randList
    global verySureList
    global verySureMean
    global randMean
    global myCount
    t = 0
    t_start = t
    terminal = False
    r_t = 0
    r_store = []
    state_store = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
    output_store = []
    critic_store = []
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    randomAct = 0
    verySure = 0
    choices = [[0, 1], [1, 0]]

    while t - t_start < t_max and terminal == False:
        t += 1
        T += 1
        intermediate_output = 0
        # action_index = 0
        # choices = np.zeros([ACTIONS])

        with graph.as_default():
            predictedChoice = model.predict(s_t)[0]
            intermediate_output = intermediate_layer_model.predict(s_t)

        # EPSILON DRIVEN
        if 1 == 1:
            # IF VERY SURE THEN USE PREDICTION
            if 0.3 < predictedChoice and predictedChoice < 0.7:
                randomAct = 0
                verySure = 1
                if 0.5 < predictedChoice:
                    a_t = [0, 1]
                else:
                    a_t = [1, 0]

            # IF NOT SURE LETS TRY RANDOM SOMETIMES, BUT LESS AND LESS
            else:
                verySure = 0
                randomChoice = np.random.rand()
                if randomChoice < epsilon:
                    randomAct = 1
                    ActionIndex = np.random.randint(0, 2)
                    a_t = choices[ActionIndex]
                else:
                    randomAct = 0
                    a_t = [0, 1] if 0.5 < predictedChoice else [1, 0]


        # DETERMINISTIC
        if 1 == 2:
            if 0.5 < predictedChoice:
                a_t = [0, 1]
            else:
                a_t = [1, 0]

        # SUCCESSFUL ORIGINAL REWRITTEN
        if 1 == 2:
            randomChoice = np.random.rand()
            if randomChoice < predictedChoice:
                a_t = [0, 1]
            else:
                a_t = [1, 0]

            # Times when random choice overwrites prediction
            if 0.5 < predictedChoice and randomChoice > predictedChoice:
                randomAct = 1
            else:
                randomAct = 0

            if predictedChoice < 0.5 and randomChoice < predictedChoice:
                randomAct = 1
            else:
                randomAct = 0

        # SUCCESSFUL ORIGINAL
        if 1 == 2:
            randomChoice = np.random.rand()
            a_t = [0, 1] if randomChoice < predictedChoice else [1, 0]  # stochastic action

        # x_t (next frame), r_t (0.1 if alive, +1.5 if it passes the pipe, -1 if it dies) and the input is a_t (action)
        x_t, r_t, terminal = game_state[thread_id].frame_step(a_t)
        x_t = preprocess(x_t)

        # SAVE FRAMES
        if 1 == 2:
            if thread_id == 0:
                mat = x_t[0, :, :, 0]
                myCount += 1

                print(mat)

                # SAVE TO CSV
                # fileName = "C:/Users/Treebeard/PycharmProjects/A3C_Keras_FlappyBird/spitout/img_" + str(myCount) + ".csv"
                # np.savetxt(fileName, mat, fmt='%2.0f', delimiter=",")

                # PLOT
                plt.imshow(mat, cmap='hot')
                #plt.show()

                fileName2 = "C:/Users/Treebeard/PycharmProjects/A3C_Keras_FlappyBird/spitout/img_" + str(myCount) + ".png"
                plt.savefig(fileName2)

        # LOG GAME STEP
        if thread_id == 0:
            randList.append(randomAct)
            verySureList.append(verySure)

            if len(randList) > 41:
                randMean = np.mean(randList[-40::])
            else:
                randMean = 0.5

            if len(verySureList) > 41:
                verySureMean = np.mean(verySureList[-40::])
            else:
                verySureMean = 0.0

            score = score + r_t

            playLog.loc[logCnt] = score, randMean, predictedChoice, verySureMean
            logCnt += 1
            if logCnt % 100 == 0:
                playLog.to_csv("playLog.csv", index=True)

            if terminal == True:
                score = 0

        with graph.as_default():
            critic_reward = model.predict(s_t)[1]

        y = 0 if a_t[0] == 1 else 1

        r_store = np.append(r_store, r_t)
        state_store = np.append(state_store, s_t, axis=0)
        output_store = np.append(output_store, y)
        critic_store = np.append(critic_store, critic_reward)

        s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)
        print(
            "Frame = " + str(T) + ", Updates = " + str(EPISODE) + ", Thread = " + str(thread_id) + ", Output = " + str(
                intermediate_output))

    if terminal == False:
        r_store[len(r_store) - 1] = critic_store[len(r_store) - 1]
    else:
        r_store[len(r_store) - 1] = -1
        s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)

    for i in range(2, len(r_store) + 1):
        r_store[len(r_store) - i] = r_store[len(r_store) - i] + GAMMA * r_store[len(r_store) - i + 1]

    # LOWER EPSILON
    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    return s_t, state_store, output_store, r_store, critic_store


# function to decrease the learning rate after every epoch. In this manner, the learning rate reaches 0, by 20,000 epochs
def step_decay(epoch):
    # print("STEP DECAY BEINGUSED----------->")
    global lrate
    decay = 3.2e-8
    lrate = LEARNING_RATE - epoch * decay
    lrate = max(lrate, 0)
    return lrate


class actorthread(threading.Thread):
    def __init__(self, thread_id, s_t):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.next_state = s_t

    def run(self):
        global episode_output
        global episode_r
        global episode_critic
        global episode_state

        threadLock.acquire()
        self.next_state, state_store, output_store, r_store, critic_store = runprocess(self.thread_id, self.next_state)
        self.next_state = self.next_state.reshape(self.next_state.shape[1], self.next_state.shape[2],
                                                  self.next_state.shape[3])

        episode_r = np.append(episode_r, r_store)
        episode_output = np.append(episode_output, output_store)
        episode_state = np.append(episode_state, state_store, axis=0)
        episode_critic = np.append(episode_critic, critic_store)

        threadLock.release()


states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))

# initializing state of each thread
for i in range(0, len(game_state)):
    image = game_state[i].getCurrentFrame()
    image = preprocess(image)
    state = np.concatenate((image, image, image, image), axis=3)
    states = np.append(states, state, axis=0)

cnt = 0
df = pd.DataFrame(columns=['reward_mean', "epsilon", "lrate", 'loss', 'policy_loss', 'critic_loss'])

while True:
    threadLock = threading.Lock()
    threads = []
    for i in range(0, THREADS):
        threads.append(actorthread(i, states[i]))

    states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))

    for i in range(0, THREADS):
        threads[i].start()

    # thread.join() ensures that all threads fininsh execution before proceeding further
    for i in range(0, THREADS):
        threads[i].join()

    for i in range(0, THREADS):
        state = threads[i].next_state
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        states = np.append(states, state, axis=0)

    e_mean = np.mean(episode_r)
    # advantage calculation for each action taken
    advantage = episode_r - episode_critic
    print("backpropagating")

    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    weights = {'o_P': advantage, 'o_V': np.ones(len(advantage))}
    # backpropagation
    history = model.fit(episode_state, [episode_output, episode_r], epochs=EPISODE + 1, batch_size=len(episode_output),
                        callbacks=callbacks_list, sample_weight=weights, initial_epoch=EPISODE)

    episode_r = []
    episode_output = []
    episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
    episode_critic = []

    # LOG SAVER
    df.loc[cnt] = e_mean, epsilon, lrate, history.history['loss'], history.history['o_P_loss'], history.history[
        'o_V_loss']
    cnt += 1
    if cnt % 100 == 0:
        df.to_csv("trainingLog.csv", index=True)

    if EPISODE % 50 == 0:
        model.save("saved_models/model_updates" + str(EPISODE))

    EPISODE += 1
    if EPISODE > 15000:
        break

print("GOT OUT")

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

BETA = 0.01
const = 1e-5


# loss function for policy output
def logloss(y_true, y_pred):  # policy loss
    return -K.sum(K.log(y_true * y_pred + (1 - y_true) * (1 - y_pred) + const), axis=-1)
    # BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term


# loss function for critic output
def sumofsquares(y_true, y_pred):  # critic loss
    return K.sum(K.square(y_pred - y_true), axis=-1)


def preprocess(image):
    image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image, (40, 40), mode='constant')
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))

    image = exposure.rescale_intensity(image, in_range=(1, 2))
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))

    image = image.reshape(1, image.shape[0], image.shape[1], 1)
    return image


game_state = game.GameState(30)

currentScore = 0
topScore = 0
a_t = [1, 0]
FIRST_FRAME = True
terminal = False
r_t = 0
myCount = 1


# -------------- code for checking performance of saved models by finding average scores for 10 runs------------------

evalGamer = pd.DataFrame(columns=['model','evalScore'])
logCnt = 0

models = [""]

fileName = "saved_models/model_updates"
modelName = 0

for i in range(1, 300):
    modelName += 50
    fileName = "saved_models/model_updates" + str(modelName)
    model = load_model(fileName, custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
    score = 0
    counter = 0

    while counter < 1:
        x_t, r_t, terminal = game_state.frame_step(a_t)
        score += 1
        if r_t == -1:
            counter += 1

        x_t = preprocess(x_t)

        if FIRST_FRAME:
            s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)

        else:
            s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

        y = model.predict(s_t)
        no = np.random.random()

        #print(y)
        if FIRST_FRAME:
            a_t = [0, 1]
            FIRST_FRAME = False
        else:
            no = np.random.rand()
            a_t = [0, 1] if no < y[0] else [1, 0]
            # a_t = [0,1] if 0.5 <y[0] else [1,0]

        if r_t == -1:
            FIRST_FRAME = True

        if score % 200 == 0:
            evalGamer.loc[logCnt] = modelName, score
            evalGamer.to_csv("evalGamer.csv", index=True)

        if terminal == True:
            print("DIED", "SCORE:", score, "Model:", modelName)
            logCnt += 1
            evalGamer.loc[logCnt] = modelName, score
            evalGamer.to_csv("evalGamer.csv", index=True)
