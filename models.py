from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline
from sklearn.model_selection import train_test_split
import scipy
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn as nn
import torch.utils.data as data
cuda = torch.device("cuda")
cpu = torch.device("cpu")
from tqdm.notebook import tqdm
from utils import get_emotion_df, get_sentence_vectors, create_triplets

def triplet_loss(anchor, positive, negative, type='abs', margin=0.2, normalize=False):
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
    

class siamese_network(nn.Module):
    def __init__(self, classes, vector_size=120, MODEL=f"cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        super().__init__()
        self.MODEL = MODEL
        self.config = AutoConfig.from_pretrained(MODEL)
        self.classes = classes
        self.vector_size = vector_size
        self.base = AutoModelForSequenceClassification.from_pretrained(self.MODEL, num_labels=self.classes, ignore_mismatched_sizes=True, is_decoder = False)

        # delete the head of the model so that we can add our own
        del self.base.classifier

        # add a shared layer for the siamese network
        self.siamese_shared_weights = self.base.roberta
        self.head1 = nn.Linear(self.config.hidden_size, vector_size)
        self.head2 = nn.Linear(self.config.hidden_size, vector_size)
        self.head3 = nn.Linear(self.config.hidden_size, vector_size)
        
    def forward(self, x1, x2, x3):
        inp1 = self.siamese_shared_weights(x1)
        inp2 = self.siamese_shared_weights(x2)
        inp3 = self.siamese_shared_weights(x3)
        inp1 = inp1[0][:, 0, :]
        inp2 = inp2[0][:, 0, :]
        inp3 = inp3[0][:, 0, :]
        # apply gelu 
        inp1 = torch.nn.functional.gelu(inp1)
        inp2 = torch.nn.functional.gelu(inp2)
        inp3 = torch.nn.functional.gelu(inp3)
        anchor = self.head1(inp1)
        positive = self.head2(inp2)
        negative = self.head3(inp3)
        
        return anchor, positive, negative

def train_siamese(train_triplets, test_triplets, siamese_model, epochs=4, print_every=25, history= {'train': [], 'test': []}, criterion=torch.nn.CrossEntropyLoss(), freeze=True):
    optimizer=torch.optim.Adam(siamese_model.parameters(), lr=0.0001)
    contrastive_loss = torch.nn.TripletMarginWithDistanceLoss(margin=siamese_model.vector_size,reduction='mean', distance_function=nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False))
    for epoch in range(epochs):
        running_loss = 0.0
        for i in tqdm(range(train_triplets.num_batches), total=train_triplets.num_batches * epochs):
            [anchor, positive, negative], anchor_class = train_triplets.get_batch()
            optimizer.zero_grad()
            output1, output2, output3 = siamese_model(anchor, positive, negative)
            loss = contrastive_loss(output1, output2, output3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i in [train_triplets.num_batches - 1, 0] or i % print_every == 0:
                print(f"Complete {epoch * train_triplets.num_batches + i + 1} / {epochs * train_triplets.num_batches}. Epoch {epoch + 1} / {epochs}.", i + 1, running_loss / print_every, end='\r')
                history['train'].append(running_loss / print_every)
                running_loss = 0.0

            # with torch.no_grad():
        #     running_loss = 0.0
        #     for i in tqdm(range(test_triplets.num_batches), total=test_triplets.num_batches * epochs):
        #         [anchor, positive, negative], anchor_class = test_triplets.get_batch()
        #         output1, output2, output3 = siamese_network_model(anchor, positive, negative)
        #         loss = contrastive_loss(output1, output2, output3)
        #         running_loss += loss.item()
        #         history_pretraining['test'].append(loss.item())
        #     print(f"Testing: Complete {epoch * test_triplets.num_batches + i + 1} / {epochs * test_triplets.num_batches}. Epoch {epoch + 1} / {epochs}.", i + 1, running_loss / test_triplets.num_batches)
    if freeze:
        for param in siamese_model.parameters():
            param.requires_grad = False
    return siamese_model, history

def train_emotion_classifier(model, ds_train, ds_test, epochs=2, print_every=25, history= {'train': [], 'test': []}, criterion=torch.nn.CrossEntropyLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    test_batches = len(ds_test)
    train_batches = len(ds_train)

    # train classify_single_input. This uses binart cross entropy loss
    for epoch in range(epochs):
        running_loss = 0.0
        i = 0 
        for batch in tqdm(ds_train, total=len(ds_train) * epochs):
            optimizer.zero_grad()
            x, y = batch
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            

            if i in [train_batches - 1, 0] or i % print_every == 0:
                print(f"Complete {epoch * train_batches + i + 1} / {epochs * train_batches}. Epoch {epoch + 1} / {epochs}.", i + 1, running_loss / print_every, end='\r')
                history['train'].append(running_loss / print_every)
                running_loss = 0.0
            i += 1

        with torch.no_grad():
            running_loss = 0.0
            for batch in tqdm(ds_test, total=len(ds_test) * epochs):
                x, y = batch
                y_pred = model(x)
                loss = criterion(y_pred, y)
                running_loss += loss.item()
                history['test'].append(loss.item())

            print(f"Complete {epoch * test_batches + i + 1 // epochs * test_batches}% Epoch {epoch + 1}/{epochs}.", "Loss",running_loss / test_batches, end='\r')
    return model, history

def prep_triplet_data(MODEL=f"cardiffnlp/twitter-xlm-roberta-base-sentiment"):
    emotion_df, classes = get_emotion_df()
    sentences = emotion_df.text.values
    sentences = get_sentence_vectors(sentences, MODEL)
    x = torch.stack([torch.tensor(sentence) for sentence in sentences]).to(cpu)

    y = emotion_df.drop('text', axis=1, inplace=False).values
    y = torch.tensor(y, dtype=torch.float32).to(cpu)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train, y_train, x_test, y_test = x_train.to(cuda), y_train.to(cuda), x_test.to(cuda), y_test.to(cuda)
    train_triplets = create_triplets(x_train, y_train, batch_size=32, shuffle=True, seed=42)
    test_triplets = create_triplets(x_test, y_test, batch_size=2, shuffle=True, seed=42)

    return classes, train_triplets, test_triplets, x_train, y_train, x_test, y_test

def prep_tensor_ds(x_train, y_train, x_test, y_test, bs=32):
    ds_train = data.TensorDataset(x_train, y_train)
    ds_train = data.DataLoader(ds_train, batch_size=bs, shuffle=True)
    ds_test = data.TensorDataset(x_test, y_test)
    ds_test = data.DataLoader(ds_test, batch_size=bs, shuffle=True)
    return ds_train, ds_test


class classify_single_input(nn.Module):
    """Inherits from the siamese network and adds a head for the classification of a single input."""
    def __init__(self, siamese_network):
        super().__init__()
        self.siamese_network = siamese_network
        self.head = nn.Linear(self.siamese_network.vector_size, self.siamese_network.classes)

    def forward(self, x):
        inp = self.siamese_network.siamese_shared_weights(x)
        inp = inp[0][:, 0, :]
        inp = torch.nn.functional.gelu(inp)
        inp = self.siamese_network.head1(inp)
        inp = self.head(inp)
        return inp


def siamese_model_dense(hp):
    """
    Set of hyperperameters and neural architechure to tune
    """
    tf.random.set_seed(0)
    x_shape, label_shape = ((300,), (29,))

    dropout = hp.Choice('dropout', [0.0, 0.05, 0.01])
    activity_regularizer_level = hp.Choice('activity_regularizer_level', [0.0, 0.001])
    bias_regularizer_level = hp.Choice('bias_regularizer_level', [0.0, 0.0001])
    kernel_regularizer_level = hp.Choice('kernel_regularizer_level', [0.0, 0.001])
    gaussian_noise = hp.Choice('gaussian_noise', [0.0, 0.01, 0.02, 0.05])
    kernel_constraint_max = hp.Choice('kernel_constraint_max', ["None", "1.0", "3.0"])
    kernel_constraint_min = hp.Choice('kernel_constraint_min', ["None", "0.05", "0.0"])
    activation = 'gelu'
    activation_first = 'selu'
    activation_head = 'gelu'
    learning_rate = 0.001
    num_layers = hp.Int('num_layers', 3, 7, step=1)
    neurons_start = hp.Int('neurons_start', 32, 256, step=16)
    neurons_middle = hp.Int('neurons_middle', 32, 256, step=16)
    neurons_end = hp.Int('neurons_end', 32, 256, step=16)


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
    activations = [activation_first] + [activation] * (num_layers - 2) + [activation_head] # create the list of activations


    inp_anchor = layers.Input(shape=x_shape)
    inp_positive = layers.Input(shape=x_shape)
    inp_negative = layers.Input(shape=x_shape)

    # create the shared weights
    shared_weights = models.Sequential()
    if gaussian_noise > 0: shared_weights.add(layers.GaussianNoise(gaussian_noise))
    for i in range(num_layers):
        if i == num_layers - 1:
            shared_weights.add(layers.BatchNormalization())
            shared_weights.add(layers.Dense(neuron_sizes[i], activation=activation_head, use_bias=True))
        else:
            if i > 0:
                if dropout > 0: shared_weights.add(layers.Dropout(dropout))
                
            shared_weights.add(layers.Dense(
                neuron_sizes[i], 
                activation=activations[i],
                use_bias=True,
                kernel_constraint=kernel_constraint,
                bias_regularizer=tf.keras.regularizers.l2(bias_regularizer_level),
                kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer_level),
                activity_regularizer=tf.keras.regularizers.l2(activity_regularizer_level)
            ))
    
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