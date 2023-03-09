from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing, callbacks, optimizers, losses, metrics
import numpy as np


def triplet_loss(dist1, dist2, dist3):
    """Calculate the triplet loss. Minimize positive_dist and maximize negative_dist."""
    positive_dist = tf.abs(dist1 - dist2)
    negative_dist = tf.abs(dist1 - dist3)
    loss = tf.reduce_mean(tf.maximum(0.0, 0.2 + positive_dist - negative_dist))
    return loss
    

def siamese_model(hp):
    x_shape, label_shape = ((300,), (29,))

    """
    Set of hyperperameters and neural architechure to tune
    """
    activation = hp.Choice('activation', ['gelu', 'swish', 'selu']) # 3 options
    activation_head = hp.Choice('activation_head', ['gelu', 'swish', 'sigmoid', 'selu']) # 4 options
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]) # 3 options
    num_layers = hp.Int('num_layers', 1, 8) # 8 options
    neurons_start = hp.Int('neurons_start', 1, 304, step=8) # 38 options
    neurons_middle = hp.Int('neurons_middle', 1, 504, step=8) #63 options
    neutons_end = hp.Int('neurons_end', 1, 504, step=8) #63 options
    # total number of options = 43436736 possible configurations
    # Thankfully, hyperband tuning cuts this number down significantly.

    """
    Generally in models, the number of neurons in each layer is increasing or decreasing (with a possible change around the midpoint of the model)
    # Here we will calculate the number of neurons in each layer and the activation function for each layer based on the trends set by the hyperparameters
    """
    neuron_sizes_1st = np.linspace(neurons_start, neurons_middle, num_layers, dtype=int) # interpolate first half of layers sizes
    neuron_sizes_2nd = np.linspace(neurons_middle, neutons_end, num_layers, dtype=int) # interpolate second half of layers sizes
    neuron_sizes = list(set(neuron_sizes_1st).union(set(neuron_sizes_2nd))) # take the union of the two sets of layer sizes
    activations = [activation] * (num_layers - 1) + [activation_head] # last layer may have different activation
    
    

    inp1 = layers.Input(shape=x_shape)
    inp2 = layers.Input(shape=x_shape)
    inp3 = layers.Input(shape=x_shape)

    # create the shared weights
    shared_weights = models.Sequential()
    for i in range(num_layers):
        shared_weights.add(layers.Dense(neuron_sizes[i], activation=activations[i]))
    
    vec1 = shared_weights(inp1)
    vec2 = shared_weights(inp2)
    vec3 = shared_weights(inp3)

    vec1_output = layers.Dense(label_shape[0], activation='softmax')(vec1)

    # create the models
    model_siamese = tf.keras.Model(inputs=[inp1, inp2, inp3], outputs=[vec1, vec2, vec3])
    model_encoder = tf.keras.Model(inputs=inp1, outputs=vec1)
    model_inference = tf.keras.Model(inputs=inp1, outputs=vec1_output)

    loss = triplet_loss(vec1, vec2, vec3)

    model_siamese.add_loss(loss)
    model_siamese.add_metric(loss, name='triplet_loss', aggregation='mean')
    model_siamese.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    model_inference.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_siamese#, model_encoder, model_inference