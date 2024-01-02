#!/bin/bash

# 変数を読み込む
source .env

# mlflowを立ち上げる
mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI}