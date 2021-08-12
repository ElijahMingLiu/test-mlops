from __future__ import absolute_import

import argparse
import os
import sys
import time

import pandas as pd
import xgboost as xgb
import pickle

# 数据被挂载到 /opt/ml/input/data/train/
# 模型输出在 /opt/ml/model/model.pkl
def train():
    print('Training started....')
    print(os.environ["SM_MODEL_DIR"])
    
    df = pd.read_csv('/opt/ml/input/data/train/iris.csv')
    clf = xgb.XGBClassifier(objective='multi:softprob')
    clf.fit(df.drop('target', axis=1), df['target'])
    print('Training finished, saving model....')
    with open('/opt/ml/model/model.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    print('Model saved.')
    
if __name__ == "__main__":
    train()