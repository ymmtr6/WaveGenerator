# coding: utf-8
import keras
from keras.models import Sequential, Model
from keras.layers import *

from keras.initializers import RandomNormal as RN, Constant


def G_model(width=120, channel=1, last_activation="relu", kernel_size=5):
    inputs_z = Input((100,), name="Z")
    in_w = int(width / 4)
    d_dim = 128
    x = Dense(in_w * d_dim, activation="tanh", name="g_dense1")(inputs_z)
    x = BatchNormalization()(x)
    x = Reshape((in_w, d_dim), input_shape=(in_w * d_dim,))(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(64, kernel_size, padding="same",
               activation="tanh", name="g_conv1")(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(1, kernel_size, padding="same",
               activation=last_activation, name="g_out")(x)
    model = Model(inputs=[inputs_z], outputs=[x], name="G")
    return model


def ACGAN_G_model(z_size=100, num_classes=6, d_dim=128, kernel_size=5):
    z = Input(shape=[z_size, ])
    class_onehot = Input(shape=[1, ], dtype="int32")
    class_embedding = Embedding(
        num_classes, z_size, embedding_initializer="glorot_normal")(class_onehot)
    h = multiply([z, class_embedding])

    x = Dense(z_size * d_dim, activation="tanh", name="g_dense1")(h)
    x = BatchNormalization()(x)
    x = Reshape((z_size, d_dim), input_shape=(z_size * d_dim,))(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(64, kernel_size, padding="same",
               activation="tanh", name="g_conv1")
    x = UpSampling1D(size=2)(x)
    x = Conv1D(1, kernel_size, padding="same",
               activation="relu", name="g_out")(x)
    model = Model(inputs=[z, class_onehot], outputs=[x], name="G")
    return model


def D_model(width=120, channel=1):
    inputs_x = Input((width, channel), name="X")
    x = Conv1D(64, 5, padding="same", activation="tanh",
               name="d_conv1")(inputs_x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 5, padding="same", activation="tanh", name="d_conv2")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu", name="d_dense1")(x)
    x = Dense(1, activation="sigmoid", name="d_out")(x)
    model = Model(inputs=[inputs_x], outputs=[x], name="D")
    return model


def ACGAN_D_model(width=100, channel=1, num_classes=6):
    input_image = Input(input_shape=(width, channel), name="X")
    x = Conv1D(64, 5, padding="same", activation="tanh",
               name="d_conv1")(inputs_image)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 5, padding="same", activation="tanh", name="d_conv2")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu", name="d_dense1")(x)

    aux = Dense(num_classes, activation="softmax", name="auxiliary")(x)
    fake = Dense(1, activation="sigmoid", name="d_out")(x)
    model = Model(inputs=[inputs_x], outputs=[fake, aux], name="D")
    return model


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model
