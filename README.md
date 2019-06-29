# NetDeviceAbnormalDetection
## File tree
- autoEncoder
  - autoEncoder.py -- train auto encoder
- clustering
  - cluster.py -- PCA clustering for visualize the input distribution
- networks
  - attention
    - create_dataset.py -- create dataset for attention framework
  - meanTeacher -- Curious AI mean teacher framework (modified)
    - dataset.py -- data loading class
    - model.py -- mean teacher framework
    - network.py -- tensorflow residual neural network v2
    - train.py -- training model
  - predict
    - create_dataset.py -- create time series dataset using time window
    - pred_regression.py -- keras residual network, training on time series dataset
    - testing.py -- 2 days ahead predicting test
  - Resnet.py -- keras residual network for present tense classification
  - adaboost.py -- adaboost for present tense classification
  - fnn.py -- fnn for present tense classification
  - one_class_model.py -- one-class framework
- toolPackage
  - draw_cm.py -- result visualization
- utils
  - evaluation.ipynb -- the whole process of data preprocessing, restore the tensorflow model and anomaly detection testing

## Present tense anomaly detection
### Data preprocessing
45 PM values -> MinMax scaler -> 45 scaled PM values -> Auto Encoder -> 75 features
Device type -> one hot -> 11 features
Concatenate the 75 features with the 11 features to form 86 features

Alarm -> 1 if there is an alarm else 0.
### Model
- tensorflow
  - meanTeacher.py
- keras
  - Resnet.py
  - one_class_model.py
  
### Evaluation
For tensorflow models, follow the flow in utils/evaluation.ipynb

## Present tense anomaly detection
Use only data with alarm.
Change the output in Present tense anomaly detection to 14 (14 alarms that we focus on)

## Anomaly prediction
### Data preprocessing
Use predict/create_dataset.py to create time series dataset.
package required:
https://github.com/Jincheng-Sun/Kylearn
### Model
A keras residual network in predict/pred_regression.py
### Evaluation
Testing in predict/testing.py
