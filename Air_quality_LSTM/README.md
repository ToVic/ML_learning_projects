Relative humidity prediction using Bidirecitonal LSTM within Tensorflow. The data is from UCI ML repository and the source is linked in the notebooks.
Basic learning rate scheduling was used when training the model.

The first modelling notebook is showing the beauty and deception of one-step predictions, as the model predicts one step ahead and then gets the actual data for the 
n+1th step.

The 'done_right' notebook is showing one possible approach of performing a multistep prediction with a model which is much simpler, yet outperforms the first complex one
fatally.
