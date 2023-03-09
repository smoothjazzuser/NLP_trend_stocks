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

    activation = hp.Choice('activation', ['relu', 'gelu', 'tanh', 'swish'])
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    num_layers = hp.Int('num_layers', 1, 6)
    layer_type = hp.Choice('layer_type', ['dense', 'conv1d', 'combined'])
    if layer_type == 'conv1d':
        kernel_size = hp.Choice('kernel_size', [3, 5, 7])
        strides = hp.Choice('strides', [1, 2, 3])
        padding = hp.Choice('padding', ['same', 'valid'])
        pool_size = hp.Choice('pool_size', [2, 3, 4])
        pool_strides = hp.Choice('pool_strides', [1, 2, 3])
        pool_padding = hp.Choice('pool_padding', ['same', 'valid'])
        max_pooling = hp.Choice('max_pooling', [True, False])

    neurons_start = hp.Int('neurons_start', 1, 300, step=32)
    neutons_end = hp.Int('neurons_end', 1, 300, step=32)
    neuron_sizes = np.linspace(neurons_start, neutons_end, num_layers, dtype=int)
    

    inp1 = layers.Input(shape=x_shape)
    inp2 = layers.Input(shape=x_shape)
    inp3 = layers.Input(shape=x_shape)

    # create the shared weights
    if layer_type == 'dense':
        shared_weights = models.Sequential()
        for i in range(num_layers):
            shared_weights.add(layers.Dense(neuron_sizes[i], activation=activation))
    elif layer_type == 'conv1d':
        shared_weights = models.Sequential()
        for i in range(num_layers):
            shared_weights.add(layers.Reshape((-1, 1)))
            shared_weights.add(layers.Conv1D(neuron_sizes[i], kernel_size, strides, padding, activation=activation))
            if max_pooling: shared_weights.add(layers.MaxPool1D(pool_size, pool_strides, pool_padding))
    elif layer_type == 'combined':
        shared_weights = models.Sequential()
        for i in range(num_layers):
            shared_weights.add(layers.Reshape((-1, 1)))
            shared_weights.add(layers.Conv1D(neuron_sizes[i], kernel_size, strides, padding, activation=activation))
            if max_pooling: shared_weights.add(layers.MaxPool1D(pool_size, pool_strides, pool_padding))
            shared_weights.add(layers.Flatten())
            shared_weights.add(layers.Dense(neuron_sizes[i], activation=activation))

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