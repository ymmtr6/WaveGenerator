# coding: utf-8

import matplotlib
import argparse
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument("--input", "-i", default="input",
                        type=str, help="csv file")
    args = parser.parse_args()
    return args


args = arg_parse()
df = pd.read_csv(args.input)
df = df.drop(columns=df.columns[[0]])

np.save(os.path.basename(args.input).split(".")[0] + ".npy", df.values)
