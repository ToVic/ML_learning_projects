Data source: https://www.kaggle.com/sakshigoyal7/credit-card-customers

Goal: predict churning customers

**Method: LightGBM and TF DNNs**

Commentary: This has been by far the most sophisticated and useful project on my journey. Within this project I have performed an exhaustive custom grid search for LightGBM
and tried out and successfully utilized Hyperband algorithm using Keras Tuner while trying to improve the predictions with a neural network model. Eventually the feature importances for both approaches have been extracted and compared.

LightGBM model was focused on general peformance and precision/recall tradeoff and Neural Network solely on Recall metric.

**LightGBM Final Performance:** precision 0.97, recall 0.64, accuracy 0.94 || precision 0.67, recall 0.87, accuracy 0.911

**DNN Final Performance:** precision 0.65, **recall 0.968**, accuracy 0.91
