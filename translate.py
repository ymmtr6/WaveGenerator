# coding: utf-8

import matplotlib
import argparse
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("503342.csv")
df = df.drop(columns=df.columns[[0]])

np.save("503342.npy", df.values)
