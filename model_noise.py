# coding: utf-8
import keras
from keras.models import Sequential, Model
from keras.layers import *

from keras.initializers import RandomNormal as RN, Constant


def G_model(width=120, channel=1, last_activation="relu", kernel_size=5):
    inputs_z = Input((100,), name="Z")
    in_w = int(width / 4)
    d_dim = 128
    x = Dense(in_w * d_dim, activation="tanh", name="d_dense1")(inputs_z)
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


def G_Deconv_model(width=120, channel=1):
    inputs_z = Input((100,), name="Z")
    in_w = int(width / 4)
    d_dim = 128
    x = Dense(in_w * d_dim, activation="tanh", name="d_dense1")(inputs_z)
    x = BatchNormalization()(x)
    x = Reshape((in_w, d_dim), input_shape=(in_w * d_dim,))(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(64, 5, padding="same", activation="tanh", name="g_conv1")(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(1, 5, padding="same", activation="relu", name="g_out")(x)
    model = Model(inputs=[inputs_z], outputs=[x], name="G")
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


def build_generator(input_size=100):
    cnn = Sequential()
    cnn.add(Dense(3 * 3 * 384, input_dim=input_size, activation="relu")
            cnn.add(Reshape((3, 3, 384))))

    # upsampling
    cnn.add(Conv2DTranspose(192, 5, strides=1, padding="valid",
                            activation="relu", kernel_initializer="glorot_normal"))
    cnn.add(BatchNormalization())

    # upsampling
    cnn.add(Conv2DTranspose(96, 5, strides=2, padding="same",
                            activaiton="relu", kernel_initializer="glorot_normal"))
    cnn.add(BatchNormalization())

    # upsampling
    cnn.add(Conv2DTranspose(1, 5, strides=2, padding="same",
                            activation="tanh", kernel_initializer="glorot_normal"))

    # this is the z space
    latent = Input(shape=(input_size,))

    # class
    image_class = Input(shape=(1,), dtype="int32")

    classes = Embedding(num_classes, input_size,
                        embeddings_initializer="glorot_normal")(image_class)

    h = multiply([input_size, classes])

    fake_image = cnn(h)

    return Model([input_size, image_class], fake_image)


def build_discriminator():
    cnn = Sequential()
    cnn.add(Conv2D(32, 3, padding="same", strides=2, input_shape=(28, 28, 1)))


"""
def G_model(width, channel=1):
    inputs = Input((100,), name="Z")
    in_w = int(width / 16)
    d_dim = 512
    x = Dense(in_w * d_dim, name="g_dense1",
              kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = Reshape((in_w, d_dim), input_shape=(in_w * d_dim))(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="q_dense1_bn")(x)
    # 1/8
    x = Conv1DTranspose(512, (5, 5), name="g_conv1", padding="same", strides=2,
                        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_conv1_bn")(x)
    # 1/4
    x = Conv1DTranspose(256, (5, 5), name="g_conv2", padding="same", strides=2,
                        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_conv2_bn")(x)
    # 1/2
    x = Conv1DTranspose(128, (5, 5), name="g_conv3", padding="same", strides=2,
                        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="g_conv3_bn")(x)
    # 1/1
    x = Conv1DTranspose(1, (5, 5), name="g_out", padding="same", strides=2, kernel_initializer=RN(
        mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Activation("tanh")(x)
    model = Model(inputs=inputs, outputs=x, name="G")
    return model


def D_model(width, channel=1):
    inputs = Input((width, channel))
    x = Conv1D(32, 5, padding="same", strides=2, name="d_conv1",
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(64, 5, padding="same", strides=2, name="d_conv2",
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(128, 5, padding="same", strides=2, name="d_conv3",
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(256, 5, padding="same", strides=2, name="d_conv4",
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid", name="d_out", kernel_initializer=RN(
        mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    model = Model(inputs=inputs, outputs=x, name="D")
    return model
"""


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model
