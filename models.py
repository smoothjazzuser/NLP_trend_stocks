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
    

def siamese_model_dense(hp):
    """
    Set of hyperperameters and neural architechure to tune
    """
    tf.random.set_seed(0)
    x_shape, label_shape = ((300,), (29,))

    batch_norm_outputs = hp.Choice('batch_norm_outputs', [True, False], default=False)
    batch_norm_layers = hp.Choice('batch_norm_layers', [True, False], default=False)
    dropout = hp.Choice('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.0)
    activity_regularizer_level = hp.Choice('activity_regularizer_level', [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0], default=0.0)
    bias_regularizer_level = hp.Choice('bias_regularizer_level', [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0], default=0.0)
    kernel_regularizer_level = hp.Choice('kernel_regularizer_level', [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0], default=0.0)
    gaussian_noise = hp.Choice('gaussian_noise', [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.0)
    kernel_constraint_max = hp.Choice('kernel_constraint_max', ["None", "1.0", "2.0", "3.0", "4.0", "5.0"], default="None")
    kernel_constraint_min = hp.Choice('kernel_constraint_min', ["None", "0.05", "0.01", "0.0", "-1.0", "-2.0"], default="None")
    activation = hp.Choice('activation', ['selu', 'gelu', 'tanh', 'sigmoid'], default='gelu')
    activation_first = hp.Choice('activation_first', ['selu', 'gelu', 'tanh', 'sigmoid'], default='selu')
    activation_head = hp.Choice('activation_head', ['selu', 'gelu', 'tanh', 'sigmoid'], default='selu')
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    num_layers = hp.Int('num_layers', 2, 10, step=1, default=7)
    neurons_start = hp.Int('neurons_start', 32, 256, step=8, default=72)
    neurons_middle = hp.Int('neurons_middle', 32, 256, step=8, default=128)
    neurons_end = hp.Int('neurons_end', 32, 256, step=8, default=128)

    if kernel_constraint_max == "None" and kernel_constraint_min == "None":
        kernel_constraint = None
    elif kernel_constraint_max == "None":
        kernel_constraint = tf.keras.constraints.MinMaxNorm(min_value=float(kernel_constraint_min), max_value=100.0)
    elif kernel_constraint_min == "None":
        kernel_constraint = tf.keras.constraints.MinMaxNorm(max_value=float(kernel_constraint_max), min_value=-100.0)
    else:
        kernel_constraint = tf.keras.constraints.MinMaxNorm(min_value=float(kernel_constraint_min), max_value=float(kernel_constraint_max))
    
    
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
            if batch_norm_outputs: shared_weights.add(layers.BatchNormalization())
            shared_weights.add(layers.Dense(neuron_sizes[i], activation=activation_head, use_bias=True))
        else:
            if i > 0:
                if dropout > 0: shared_weights.add(layers.Dropout(dropout))
                if gaussian_noise > 0: shared_weights.add(layers.GaussianNoise(gaussian_noise))
                if batch_norm_layers: shared_weights.add(layers.BatchNormalization())
            shared_weights.add(layers.Dense(
                neuron_sizes[i], 
                activation=activations[i],
                use_bias=True,
                kernel_constraint=kernel_constraint,
                bias_regularizer=tf.keras.regularizers.l2(bias_regularizer_level),
                kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer_level),
                activity_regularizer=tf.keras.regularizers.l2(activity_regularizer_level)
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