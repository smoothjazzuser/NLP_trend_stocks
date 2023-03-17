from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, preprocessing, callbacks, optimizers, losses, metrics
import numpy as np


def triplet_loss(anchor, positive, negative, type='contrastive', margin=0.2, normalize=False):
    """Calculate the triplet loss. Minimize positive_dist and maximize negative_dist."""
    if normalize:
        max_ = tf.math.maximum(tf.math.maximum(tf.reduce_max(anchor), tf.reduce_max(positive)), tf.reduce_max(negative))
        min_ = tf.math.minimum(tf.math.minimum(tf.reduce_min(anchor), tf.reduce_min(positive)), tf.reduce_min(negative))
        anchor = (anchor - min_) / (max_ - min_)
        positive = (positive - min_) / (max_ - min_)
        negative = (negative - min_) / (max_ - min_)

    if type == 'abs':
        positive_dist =  tf.reduce_sum(tf.abs(anchor - positive), axis=1)
        negative_dist =  tf.reduce_sum(tf.abs(anchor - negative), axis=1)    
        loss = tf.maximum(0.0, margin + positive_dist - negative_dist)

    elif type == 'square':
        positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        loss = tf.maximum(0.0, margin + positive_dist - negative_dist)

    elif type == 'contrastive':
        pos_cos = losses.cosine_similarity(anchor, positive)
        neg_cos = losses.cosine_similarity(anchor, negative)
        dist = tf.reduce_sum(neg_cos - pos_cos)

        # In this function, a higher x means a higher loss. Negatives tend towards a limit of zero
        loss = tf.math.log(1 + tf.exp(dist))

    return tf.reduce_mean(loss)
    

def siamese_model_dense(hp):
    """
    Set of hyperperameters and neural architechure to tune
    """
    tf.random.set_seed(0)
    x_shape, label_shape = ((300,), (29,))

    batch_norm_outputs = hp.Choice('batch_norm_outputs', [True, False], default=False)
    batch_norm_layers = hp.Choice('batch_norm_layers', [False, True], default=False)
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
    # batch_norm_outputs = True
    # batch_norm_layers = False
    # dropout = 0.0
    # activity_regularizer_level = 0.0
    # bias_regularizer_level = 0.0
    # kernel_regularizer_level = 0.0
    # gaussian_noise = 0.0
    # kernel_constraint_max = "None"
    # kernel_constraint_min = "None"
    # activation = 'gelu'
    # activation_first = 'selu'
    # activation_head = 'selu'
    # learning_rate = 1e-6
    # num_layers = 7
    # neurons_start = 72
    # neurons_middle = 128
    # neurons_end = hp.Choice('neurons_end', [32, 64])

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


    inp_anchor = layers.Input(shape=x_shape)
    inp_positive = layers.Input(shape=x_shape)
    inp_negative = layers.Input(shape=x_shape)

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
            shared_weights.add(layers.Dense(
                neuron_sizes[i], 
                activation=activations[i],
                use_bias=True,
                kernel_constraint=kernel_constraint,
                bias_regularizer=tf.keras.regularizers.l2(bias_regularizer_level),
                kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer_level),
                activity_regularizer=tf.keras.regularizers.l2(activity_regularizer_level)
            ))
            if batch_norm_layers: 
                shared_weights.add(layers.BatchNormalization())
    
    vec_anchor = shared_weights(inp_anchor)
    vec_positive = shared_weights(inp_positive)
    vecinp_negative = shared_weights(inp_negative)

    vec_anchor_output = layers.Dense(label_shape[0], activation='softmax')(vec_anchor)

    # create the models
    model_siamese = tf.keras.Model(inputs=[inp_anchor, inp_positive, inp_negative], outputs=[vec_anchor, vec_positive, vecinp_negative])
    model_encoder = tf.keras.Model(inputs=inp_anchor, outputs=vec_anchor)
    model_inference = tf.keras.Model(inputs=inp_anchor, outputs=vec_anchor_output)

    loss = triplet_loss(vec_anchor, vec_positive, vecinp_negative)

    model_siamese.add_loss(loss)
    model_siamese.add_metric(loss, name='triplet_loss', aggregation='mean')
    model_siamese.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    model_inference.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_siamese#, model_encoder, model_inference