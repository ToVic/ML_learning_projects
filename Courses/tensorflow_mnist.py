import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info = True, as_supervised=True)
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

#define scaling function (256 shades of gray with values 0-255)

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

#shuffling the data before creating a validation dataset

BUFFER_SIZE = 10000 #if bigger than num_samples, shuffling happens uniformly at once
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
BATCH_SIZE = 100
#we ALWAYS define hyperparameters as a variable
#batch size = SGD
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

### OUTLINE THE MODEL ###

input_size = 784 # 28 times 28
output_size = 10 #one for each digit
hidden_layer_size = 200

model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape = (28,28,1)),
                            #create a vector out of a tensor
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            # dot product of the inputs and the weights and adds bias
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            #we can insert more layers, but whats the right depth?
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                            tf.keras.layers.Dense(output_size, activation = 'softmax')
                            #final layer must transform output into probabilities
                            ])
### SPECIFY OPTIMIZER AND LOSS ###

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


### TRAINING ###

NUM_EPOCHS = 5

model.fit(train_data, epochs = NUM_EPOCHS, validation_data=(validation_inputs, validation_targets),\
          verbose = 2)

### TESTING ###
test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100))
#Test loss: 0.07. Test accuracy: 97.91%
#after testing you cant fiddle with the hyperparameters anymore, since the model has already seen/
#your test data
