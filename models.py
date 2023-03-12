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
    """
    Set of hyperperameters and neural architechure to tune
    """
    x_shape, label_shape = ((300,), (29,))

    activation = 'gelu'
    activation_first = 'selu'
    activation_head = 'gelu'
    learning_rate = 0.001
    num_layers = 7
    neurons_start = hp.Int('neurons_start', 72 - 24, 72 + 24, step=2) #72
    neurons_middle = hp.Int('neurons_middle', 168 - 42, 168 + 42, step=2) # 168
    neurons_end = hp.Int('neurons_end', 224 - 42, 224 + 42, step=2) # 224

    """
    Generally in models, the number of neurons in each layer is increasing or decreasing (with a possible change around the midpoint of the model)
    # Here we will calculate the number of neurons in each layer and the activation function for each layer based on the trends set by the hyperparameters
    """
    neuron_sizes_1st = np.linspace(neurons_start, neurons_middle, num_layers, dtype=int) # interpolate first half of layers sizes
    neuron_sizes_2nd = np.linspace(neurons_middle, neurons_end, num_layers, dtype=int) # interpolate second half of layers sizes
    neuron_sizes = list(set(neuron_sizes_1st).union(set(neuron_sizes_2nd))) # take the union of the two sets of layer sizes

    activations = [activation_first] + [activation] * (num_layers - 2)


    inp1 = layers.Input(shape=x_shape)
    inp2 = layers.Input(shape=x_shape)
    inp3 = layers.Input(shape=x_shape)

    # create the shared weights
    shared_weights = models.Sequential()
    for i in range(num_layers):
        if i == num_layers - 1:
            shared_weights.add(layers.Dense(neuron_sizes[i], activation=activation_head, use_bias=True))
        else:
            shared_weights.add(layers.Dense(
                neuron_sizes[i], 
                activation=activations[i],
                use_bias=True,
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