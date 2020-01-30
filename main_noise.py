from keras.datasets import mnist
from model_noise import *
import config_noise as cf
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import glob
import numpy as np
import csv
import argparse
from keras import backend as K
import tensorflow as tf
import keras
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
data_minimum = 0
data_maximum = 100

np.random.seed(cf.Random_seed)

Width = 120
Channel = 1
input_file_name = "503342.npy"


class Main_train():
    def __init__(self):
        pass

    def train(self):
        # Load network model
        g = G_model(width=Width, channel=Channel)
        d = D_model(width=Width, channel=Channel)
        c = Combined_model(g=g, d=d)

        g.summary()

        # Set Optimizers (Adam, lr=0.0002, beta1_0.5)
        g_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        # model compile(loss=binary_crosentropy)
        # Freeze Layers
        g.compile(loss='binary_crossentropy', optimizer='SGD')
        d.trainable = False
        for layer in d.layers:
            layer.trainable = False
        c.compile(loss='binary_crossentropy', optimizer=g_opt)
        d.trainable = True
        for layer in d.layers:
            layer.trainable = True
        d.compile(loss='binary_crossentropy', optimizer=d_opt)

        # Prepare Training data
        X_train, data_minimum, data_maximum = generate_X_train(input_file_name)
        train_num = X_train.shape[0]
        train_num_per_step = train_num // cf.Minibatch
        data_inds = np.arange(train_num)
        max_ite = cf.Minibatch * train_num_per_step

        # Start Train
        print('-- Training Start!!')
        if cf.Save_train_combine is None:
            print("generated image will not be stored")
        elif cf.Save_train_combine is True:
            print("generated image write combined >>", cf.Save_train_img_dir)
        elif cf.Save_train_combine is False:
            print("generated image write separately >>", cf.Save_train_img_dir)

        fname = os.path.join(cf.Save_dir, 'loss.txt')
        f = open(fname, 'w')
        f.write("Iteration,G_loss,D_loss{}".format(os.linesep))

        for ite in range(cf.Iteration):
            ite += 1
            # Discremenator training
            train_ind = ite % (train_num_per_step - 1)
            if ite % (train_num_per_step + 1) == max_ite:
                np.random.shuffle(data_inds)

            _inds = data_inds[train_ind *
                              cf.Minibatch: (train_ind+1) * cf.Minibatch]
            x_real = X_train[_inds]

            z = np.random.uniform(-1, 1, size=(cf.Minibatch, 100))
            # input_noise = np.random.normal(0, 0.3, size=(cf.Minibatch, 100))
            x_fake = g.predict([z], verbose=0)
            x = np.concatenate((x_real, x_fake))
            t = [1] * cf.Minibatch + [0] * cf.Minibatch
            d_loss = d.train_on_batch(x, t)

            # Generator training
            z = np.random.uniform(-1, 1, size=(cf.Minibatch, 100))
            # input_noise = np.random.normal(0, 0.3, size=(cf.Minibatch, 100))
            g_loss = c.train_on_batch([z], [1] * cf.Minibatch)

            con = '|'
            if ite % cf.Save_train_step != 0:
                for i in range(ite % cf.Save_train_step):
                    con += '>'
                for i in range(cf.Save_train_step - ite % cf.Save_train_step):
                    con += ' '
            else:
                for i in range(cf.Save_train_step):
                    con += '>'
            con += '| '
            con += "Ite:{}, g: {:.6f}, d: {:.6f} ".format(ite, g_loss, d_loss)
            sys.stdout.write("\r"+con)

            if ite % cf.Save_train_step == 0 or ite == 1:
                print()
                f.write("{},{},{}{}".format(ite, g_loss, d_loss, os.linesep))
                # save weights
                d.save_weights(cf.Save_d_path)
                g.save_weights(cf.Save_g_path)

                gerated = g.predict([z], verbose=0)
                # save some samples
                if cf.Save_train_combine is True:
                    # write_csv(gerated, index=ite,
                    #            dir_path=cf.Save_train_img_dir)
                    write_graph(gerated, index=ite,
                                dir_path=cf.Save_train_img_dir)
                elif cf.Save_train_combine is False:
                    save_images_separate(
                        gerated, index=ite, dir_path=cf.Save_train_img_dir)
        f.close()
        # Save trained model
        write_csv(gerated, index=ite, dir_path=cf.Save_train_img_dir)
        d.save_weights(cf.Save_d_path)
        g.save_weights(cf.Save_g_path)
        print('Model saved -> ', cf.Save_d_path, cf.Save_g_path)


class Main_test():
    def __init__(self):
        pass

    def test(self):
        """
        Test Functions
        """
        # Load network model
        g = G_model(width=Width, channel=Channel)
        g.load_weights(cf.Save_g_path, by_name=True)

        print('-- Test start!!')
        if cf.Save_test_combine is None:
            print("generated image will not be stored")
        elif cf.Save_test_combine is True:
            print("generated image write combined >>", cf.Save_test_img_dir)
        elif cf.Save_test_combine is False:
            print("generated image write separately >>", cf.Save_test_img_dir)
        pbar = tqdm(total=cf.Test_num)

        for i in range(cf.Test_num):
            input_noise = np.random.uniform(-1,
                                            1, size=(cf.Test_Minibatch, 100))
            g_output = g.predict(input_noise, verbose=0)

            if cf.Save_test_combine is True:
                write_csv(g_output, index=i, dir_path=cf.Save_test_img_dir)
                write_graph(g_output, index=i,
                            dir_path=cf.Save_test_img_dir)
            elif cf.Save_test_combine is False:
                save_images_separate(
                    g_output, index=i, dir_path=cf.Save_test_img_dir)
            pbar.update(1)


def write_csv(imgs, index, dir_path):
    """
    np.array(batch, width, channel)
    with value range [0, 1]
    """
    B, W, C = imgs.shape
    imgs = denormalize(imgs, minimum=data_minimum,
                       maximum=data_maximum)
    fname = str(index).zfill(len(str(cf.Iteration))) + '.csv'
    save_path = os.path.join(dir_path, fname)
    # numpyの機能で保存する(channel次元は削除する)
    np.savetxt(save_path, np.reshape(imgs, [B, W]).astype(
        np.uint8), fmt="%d", delimiter=",")


def save_images_separate(imgs, index, dir_path):
    pass


def write_graph(imgs, index, dir_path):
    """

    """
    B, W, C = imgs.shape
    imgs = denormalize(imgs, minimum=data_minimum, maximum=data_maximum,)
    fname = str(index).zfill(len(str(cf.Iteration))) + '.png'
    save_path = os.path.join(dir_path, fname)
    imgs = np.reshape(imgs, [B, W]).astype(np.uint8)
    x = np.linspace(0, 24, 120)
    plt.figure(figsize=(10, 10))
    for i, d in enumerate(imgs):
        plt.subplot(4, 4, i + 1)
        plt.xticks([0, 6, 12, 18, 24])
        plt.tick_params(length=0)
        plt.xlim([0, 24])
        plt.ylim([0, max(40, d.max())])
        plt.plot(x, d)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument("--original", dest="original", action="store_true")
    args = parser.parse_args()
    return args


def generate_X_train(f_name):
    """
    学習用データの作成
    画像に戻すときのために，min,maxを返す．
    """
    d = np.load(f_name)
    # 0が続くデータは削除
    d = d[~np.all(d < 5, axis=1)]
    # 2σを基準に外れ値を除去
    d = outlier(d)
    # [0, 1]に正規化
    d, (minimum, maximum) = normalize(d)
    # 次元拡張(チャンネル数入力の都合)
    X_train = d[:, :, None]
    X_train = X_train.astype(np.float32)
    print("{}: {}".format(f_name, X_train.shape[0]))
    return X_train, minimum, maximum


def outlier(x):
    """
    外れ値の削除: 2σの範囲から離れる場合
    """
    average = np.mean(x)
    sd = np.std(x)
    outlier_min = max(0, average - sd * 2)
    outlier_max = average + sd * 2
    x = x[np.all(outlier_min <= x, axis=1)]
    x = x[np.all(x <= outlier_max, axis=1)]
    return x


def normalize(x, axis=None):
    """
    正規化(min-max)
    """
    minimum = x.min(axis=axis)
    maximum = x.max(axis=axis)
    return (x - minimum) / (maximum - minimum), (minimum, maximum)


def denormalize(x, minimum, maximum):
    """
    正規化(min-max)した値を元に戻す．
    """
    return x * (maximum - minimum) + minimum


if __name__ == '__main__':
    # train, test, originalのフラグにそれぞれ対応する
    args = arg_parse()

    if args.train:
        main = Main_train()
        main.train()
    if args.test:
        main = Main_test()
        main.test()
    if args.original:
        X_train, data_minimum, data_maximum = generate_X_train(input_file_name)
        os.makedirs("./original", exist_ok=True)
        write_csv(X_train, index=0, dir_path="./original/")
        np.random.shuffle(X_train)
        for i in range(X_train.shape[0] // 16 - 1):
            start = i * 16
            end = (i+1) * 16
            print(start, end, i)
            write_graph(X_train[start:end], i,
                        "./original")

    if not (args.train or args.test or args.original):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
