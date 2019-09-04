import numpy as np
import xgboost as xgb
X = np.load('data/feature_train.npy')
y = np.load('data/target_binary_train.npy')
classifier = xgb.XGBClassifier(objective="binary:logistic", random_state=22)
classifier.fit(X, y)