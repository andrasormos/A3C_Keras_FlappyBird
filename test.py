import numpy as np
import sys

sys.path.append("game/")

import pygame
import wrapped_flappy_bird as game

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt

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
    image = skimage.transform.resize(image, (84, 84), mode='constant')
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
modelName = 6000

for i in range(1, 120):
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



