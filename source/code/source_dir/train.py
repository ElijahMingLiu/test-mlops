from __future__ import absolute_import

import argparse
import os
import sys
import time
import pandas as pd

# from utils import print_files_in_path, save_model_artifacts

# 数据被挂载到 /opt/ml/input/data/train/

def train():
    print('Finally!! Run!!!')
    # df = pd.read_csv('/opt/ml/input/data/666/dummy.csv')
    # print(df)
    print(os.environ["SM_MODEL_DIR"])
if __name__ == "__main__":
    train()
