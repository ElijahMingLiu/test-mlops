from __future__ import absolute_import

import argparse
import os
import sys
import time
import pandas as pd

from utils import print_files_in_path, save_model_artifacts

# 数据被挂载到 /opt/ml/input/data/train/

def train():
    print('zzzzwwww')
    df = pd.read_csv('/opt/ml/input/data/666/dummy.csv')
    print(df)

if __name__ == "__main__":
    train()
    print_files_in_path(os.environ["SM_CHANNEL_TRAIN"])
