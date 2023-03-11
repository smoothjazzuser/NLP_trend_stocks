from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
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
    
    activity_regularizer = hp.Choice('activity_regularizer', ['None', 'l1', 'l2'])
    weight_regularizer = hp.Choice('weight_regularizer', ['None', 'l1', 'l2'])
    kernel_regularizer = hp.Choice('kernel_regularizer', ['None', 'l1', 'l2'])
    activation = hp.Choice('activation',             ['gelu', 'selu', 'tanh', 'softplus', 'swish'])
    activation_first = hp.Choice('activation_first', ['gelu', 'selu', 'tanh', 'softplus', 'swish']) 
    activation_head = hp.Choice('activation_last',   ['gelu', 'selu', 'tanh', 'softplus', 'swish'])  
    learning_rate = 0.001
    num_layers = 7
    neurons_start = hp.Int('neurons_start', 64, 120, step=4) # steps based in the range of the best hyperparameters from a previous courser run
    neurons_middle = hp.Int('neurons_middle', 120, 240, step=4)
    neurons_end = hp.Int('neurons_end', 120, 300, step=4)

    """
    Generally in models, the number of neurons in each layer is increasing or decreasing (with a possible change around the midpoint of the model)
    # Here we will calculate the number of neurons in each layer and the activation function for each layer based on the trends set by the hyperparameters
    """
    neuron_sizes_1st = np.linspace(neurons_start, neurons_middle, num_layers, dtype=int) # interpolate first half of layers sizes
    neuron_sizes_2nd = np.linspace(neurons_middle, neurons_end, num_layers, dtype=int) # interpolate second half of layers sizes
    neuron_sizes = list(set(neuron_sizes_1st).union(set(neuron_sizes_2nd))) # take the union of the two sets of layer sizes
    activations = [activation_first] + [activation] * (num_layers - 2) + [activation_head] 
    if activity_regularizer != 'None':
        activity_regularizer = getattr(keras.regularizers, activity_regularizer)
    if weight_regularizer != 'None':
        weight_regularizer = getattr(keras.regularizers, weight_regularizer)
    if kernel_regularizer != 'None':
        kernel_regularizer = getattr(keras.regularizers, kernel_regularizer)
    activity_regularizer = [activity_regularizer] * (num_layers - 1) + [None]
    weight_regularizer = [weight_regularizer] * (num_layers - 1) + [None]
    kernel_regularizer = [kernel_regularizer] * (num_layers - 1) + [None]


    inp1 = layers.Input(shape=x_shape)
    inp2 = layers.Input(shape=x_shape)
    inp3 = layers.Input(shape=x_shape)

    # create the shared weights
    shared_weights = models.Sequential()
    for i in range(num_layers):
        shared_weights.add(layers.Dense(
            neuron_sizes[i], 
            activation=activations[i],
            activity_regularizer=activity_regularizer[i],
            weight_regularizer=weight_regularizer[i],
            kernel_regularizer=kernel_regularizer[i]
        ))
    
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