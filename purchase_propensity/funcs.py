import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

def data_split(df, sizes = 30000, set_index = True):
    if set_index:
        cols = list(df.columns)
        df.set_index(cols[0], inplace=True)
    X, y = df.iloc[:,:-1], df.iloc[:,-1:]
    X, y = np.asarray(X).astype(np.float64), np.asarray(y).astype(np.float64)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = sizes)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size = sizes)
    for s,t in zip([X_train, y_train, X_valid, y_valid, X_test, y_test],['Training data: ','Training targets: ','Validation data: ',
                                                                      'Validation targets: ','Test data: ','Test targets: ']):
        print(t, len(s), ' rows returned.')
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
              
def train_model(X, y, Xv, yv, lr=0.001, epochs = 30, batch_size = 32, patience=2, class_w = None):
    model = keras.Sequential([
        keras.layers.Dense(48, input_shape=(X.shape[1],), activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(48, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    adam = keras.optimizers.Adam(lr)
    
    ES = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights = True, monitor='val_loss')
    
    model.compile(loss='BinaryCrossentropy', metrics = 'AUC', optimizer = adam)
    
    history = model.fit(X, y, batch_size = batch_size, epochs = epochs, verbose=2, validation_data=(Xv, yv),
                        class_weight = class_w, callbacks = [ES])
    
    return history, model


def plot_cm(model, data, labels, p=0.5):
    predictions = model.predict(data)
    
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize = (5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion matrix at probability: {:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    print('True Negatives: ',tn)
    print('False Positives: ',fp)
    print('False Negatives: ',fn)
    print('True Positives: ',tp)
    print('Total predicted purchases: ', tp+fp)
    
    print('Model precision: ', (tp/(tp+fp).round(3)))
    print('Model recall: ', (tp/(tp+fn).round(3)))
    print('Model accuracy: ', ((tp+tn)/(tp+tn+fp+fn).round(3)))

def plot_learn(history):
    plt.figure(figsize=(15,7))
    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
