from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn as nn
import torch.utils.data as data
cuda = torch.device("cuda")
cpu = torch.device("cpu")
from tqdm.notebook import tqdm
from utils import get_emotion_df, create_triplets
import random
import copy
import contextlib

class siamese_network(nn.Module):
    """Instantiate the xlm-roberta-base-sentiment model and add a siamese network head to it to replace the sentiment classifier head with emotion classifier head. Allowes for the model to later be altered to train on non-triplet data after pre-training on triplet data.
    
    Arguments:
        classes {int} -- The number of classes to classify into
        vector_size {int} -- The size of the vector to output from the siamese network. This is the size of the vector that will be used to train the emotion classifier.
        model {str} -- The pretrained model to use. Defaults to "cardiffnlp/twitter-xlm-roberta-base-sentiment"."""
    def __init__(self, classes, vector_size=120, MODEL=f"cardiffnlp/twitter-xlm-roberta-base-sentiment", head=False):
        super().__init__()
        self.MODEL = MODEL
        self.head = head
        self.config = AutoConfig.from_pretrained(MODEL, num_labels=classes, is_decoder = False, fine_tuning_task = 'text-classification')
        self.classes = classes
        self.vector_size = vector_size
        self.base = AutoModelForSequenceClassification.from_pretrained(self.MODEL, ignore_mismatched_sizes=True, is_decoder = False) #num_labels=self.classes, 
        # delete the head of the model so that we can add our own
        del self.base.classifier

        # add a shared layer for the siamese network
        self.siamese_shared_weights = self.base.roberta
        self.head1 = nn.Linear(self.config.hidden_size, vector_size)
        self.alphadropout = nn.AlphaDropout(p=0.1)
        
    def forward(self, x1, x2, x3):
        inp1 = self.siamese_shared_weights(x1)
        inp2 = self.siamese_shared_weights(x2)
        inp3 = self.siamese_shared_weights(x3)
        inp1 = inp1[0][:, 0, :]
        inp2 = inp2[0][:, 0, :]
        inp3 = inp3[0][:, 0, :]
        if self.head:
            inp1 = self.alphadropout(inp1)
            inp2 = self.alphadropout(inp2)
            inp3 = self.alphadropout(inp3)
            anchor = self.head1(inp1)
            positive = self.head1(inp2)
            negative = self.head1(inp3)
        else:
            anchor = inp1
            positive = inp2
            negative = inp3

        anchor = torch.nn.functional.softmax(anchor, dim=1)
        positive = torch.nn.functional.softmax(positive, dim=1)
        negative = torch.nn.functional.softmax(negative, dim=1)
        
        return anchor, positive, negative

def pre_train_using_siamese(train_triplets, test_triplets, siamese_model, classes, epochs=4, print_every=500, history= {'train': [], 'test': []}, criterion=None, early_stopping=True, patience=100):
    """Pret-train the emotion classifier (the weights prior to the final layer) using a siamese network in order to ensure the classifier has an easier job.
    
    Triplet test is currenetly not used due to a bug somewhere in the code.
    
    Arguments:
        train_triplets {Triplets} -- The training triplets generator
        test_triplets {Triplets} -- The testing triplets generator
        siamese_model {nn.Module} -- The siamese network model
        epochs {int} -- An estimate of the number of epochs to run for... this is not exact due to the combinatorial nature of the generator. Not all of the data will be seen after one epoch, and future epochs will likily output different combinations of data (although a set seed is used).
        print_every {int} -- How often to print the loss
        history {dict} -- A dictionary to store the loss history
        criterion {nn.Module} -- The loss function
        freeze {bool} -- Whether to freeze the weights of the siamese network after training

    Returns:
        nn.Module -- The siamese network model
        history -- The loss history dictionary
        """
    bs = train_triplets.batch_size
    siamese_model.train()
    for param in siamese_model.base.parameters():
        param.requires_grad = True
        
    if siamese_model.head == False:
        for name, param in siamese_model.base.roberta.encoder.layer[11].named_parameters():
            param.requires_grad = True
        for name, param in siamese_model.base.roberta.encoder.layer[10].named_parameters():
            param.requires_grad = True

    print(f'trainable: {[name for name, param in siamese_model.named_parameters() if param.requires_grad]}')
    if early_stopping:
        best_loss = np.inf
        patience_counter = 0

    optimizer=torch.optim.Adam(siamese_model.parameters(), lr=0.001)
    if criterion is None:
        contrastive_loss = torch.nn.TripletMarginWithDistanceLoss(margin=1, swap=False, reduction='sum', distance_function=torch.nn.PairwiseDistance(p=1, eps=1e-06, keepdim=False))
    else:
        contrastive_loss = criterion
    with tqdm(total=epochs * train_triplets.num_batches) as pbar:
        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(train_triplets.num_batches):
                [anchor, positive, negative], anchor_class = train_triplets.get_batch()
                optimizer.zero_grad()
                output1, output2, output3 = siamese_model(anchor, positive, negative)
                loss = contrastive_loss(output1, output2, output3)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (i in [train_triplets.num_batches - 1] or i % print_every == 0) and i != 0:
                    with torch.no_grad():
                        # turn off dropout, but only for this loop
                        siamese_model.eval()
                        valid_loss = 0.0
                        for i in range(print_every):
                            [anchor, positive, negative], anchor_class = test_triplets.get_batch()
                            output1, output2, output3 = siamese_model(anchor, positive, negative)
                            loss = contrastive_loss(output1, output2, output3)
                            valid_loss += loss.item()
                        history['test'].append(valid_loss / print_every * bs)
                        siamese_model.train()
                        
                    history['train'].append(running_loss / print_every)
                    running_loss = 0.0
                    pbar.desc = f"train loss: {history['train'][-1]}, val loss: {history['test'][-1]}, patience {patience - patience_counter}/{patience}"
                    pbar.update(print_every)

                    if not early_stopping:
                        continue

                    if history['test'][-1] == 0.0:
                        best_weights = copy.deepcopy(siamese_model.state_dict())
                    if history['test'][-1] < best_loss:
                        best_loss = history['test'][-1]
                        best_weights = copy.deepcopy(siamese_model.state_dict())
                        patience_counter = 0
                    elif patience_counter > patience:
                        print(f'Early stopping: restoring best weights with test loss {best_loss}')
                        siamese_model.load_state_dict(best_weights)
                        return siamese_model, history
                    elif history['test'][-1] > best_loss:
                        patience_counter += 1
                    elif float(history['test'][-1]) == 0.0:
                        patience_counter += 0.1
                    elif history['test'][-1] in [np.inf, np.nan]:
                        print('Early stopping: NaN or inf loss')
                        siamese_model.load_state_dict(best_weights)
                        return siamese_model, history

        if early_stopping:
            print(f'Early stopping: restoring best weights with test loss {best_loss}')
            siamese_model.load_state_dict(best_weights)
        siamese_model.eval()  
    return siamese_model, history

def test_siamese_model(model, test_triplets, print_every=100, history= {'test': []}, criterion=None):
    """Test the siamese network model
    
    Arguments:
        model {nn.Module} -- The siamese network model
        test_triplets {Triplets} -- The testing triplets generator
        print_every {int} -- How often to print the loss
        history {dict} -- A dictionary to store the loss history
        criterion {nn.Module} -- The loss function

    Returns:
        nn.Module -- The siamese network model
        history -- The loss history dictionary
        """
    if criterion == None:
        contrastive_loss = torch.nn.TripletMarginWithDistanceLoss(margin=model.vector_size,reduction='mean', distance_function=nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False))
    else:
        contrastive_loss = criterion

    with torch.no_grad():
        valid_loss = 0.0
        for i in range(test_triplets.num_batches):
            [anchor, positive, negative], anchor_class = test_triplets.get_batch()
            output1, output2, output3 = model(anchor, positive, negative)
            loss = contrastive_loss(output1, output2, output3)
            valid_loss += loss.item()
            if i in [test_triplets.num_batches - 1, 0] or i % print_every == 0:
                print(f"Complete {i + 1} / {test_triplets.num_batches}.", valid_loss / print_every, end='\r')
                history['test'].append(valid_loss / print_every)
                valid_loss = 0.0

    # print mean loss over the entire test set
    print(f"Test loss: {np.mean(history['test'])}")
    return history

def test_emotion_classifier(model, ds_test, print_every=100, history= {'test': []}, criterion=torch.nn.CrossEntropyLoss()):
    """Test the emotion classifier model
    
    Arguments:
        model {nn.Module} -- The emotion classifier model
        ds_test {torch.utils.data.Dataset} -- The testing dataset
        print_every {int} -- How often to print the loss
        history {dict} -- A dictionary to store the loss history
        criterion {nn.Module} -- The loss function

    Returns:
        nn.Module -- The emotion classifier model
        history -- The loss history dictionary
        """
    with torch.no_grad():
        valid_loss = 0.0
        for i in range(len(ds_test)):
            anchor, anchor_class = ds_test[i]
            output = model(anchor)
            loss = criterion(output, anchor_class)
            valid_loss += loss.item()
            if i in [len(ds_test) - 1, 0] or i % print_every == 0:
                print(f"Complete {i + 1} / {len(ds_test)}.", valid_loss / print_every, end='\r')
                history['test'].append(valid_loss / print_every)
                valid_loss = 0.0

    # print mean loss over the entire test set
    print(f"Test loss: {np.mean(history['test'])}")
    return history

def train_emotion_classifier(model, ds_train, ds_test, epochs=2, print_every=500, history= {'train': [], 'test': []}, early_stopping=True, patience=100, criterion=torch.nn.CrossEntropyLoss()):
    """Train the emotion classifier using the siamese network weights as a starting point. This only updates the final layer of the model.
    
    Arguments:
        model {nn.Module} -- The emotion classifier model (can be pretrained or not)
        ds_train {torch.utils.data.Dataset} -- The training dataset
        ds_test {torch.utils.data.Dataset} -- The testing dataset
        epochs {int} -- The number of epochs to train for
        print_every {int} -- How often to print the loss
        history {dict} -- A dictionary to store the loss history
        criterion {nn.Module} -- The loss function

    Returns:
        nn.Module -- The emotion classifier model
        history -- The loss history dictionary
        """

    assert print_every > 3 and print_every < len(ds_train), "print_every must be between 3 and the length of the dataset"
    assert epochs > 0, "epochs must be greater than 0"
    assert len(ds_train) > 0, "ds_train must have at least one element"
    assert len(ds_test) > 0, "ds_test must have at least one element"
    
    model.train()
    for param in model.siamese_network.parameters():
            param.requires_grad = False
    model.head.requires_grad = True
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True
    model.siamese_network.head1.requires_grad = True
    #for name, param in model.siamese_network.base.roberta.encoder.layer[11].named_parameters():
        #param.requires_grad = True

    print(f'trainable: {[name for name, param in model.named_parameters() if param.requires_grad]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    test_batches = len(ds_test)
    train_batches = len(ds_train)
    
    patience_ = patience
    if early_stopping:
        best_loss = np.inf

    with tqdm(total=epochs * len(ds_train)) as pbar:
        for epoch in range(epochs):
            running_loss = 0.0
            valid_loss = 0.0
            batch_num = 0 
            for batch in ds_train:
                optimizer.zero_grad()
                x, y = batch
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if batch_num in [train_batches - 1, 0] or batch_num % print_every == 0:
                    with torch.no_grad():
                        model.eval()
                        valid_loss = 0.0
                        # validation set random batch of size print_every, 
                        for _ in range(print_every//2):
                            x, y = next(iter(ds_test))
                            y_pred = model(x)
                            loss = criterion(y_pred, y)
                            valid_loss += loss.item()
                        history['test'].append(valid_loss / (print_every//2))
                        model.train()
                    history['train'].append(running_loss / print_every)
                    running_loss = 0.0
                batch_num += 1
                pbar.desc = f"train loss: {history['train'][-1]}, val loss: {history['test'][-1]}, patience {patience_}/{patience}"
                pbar.update(1)

                
                if not early_stopping:
                    continue

                if batch_num % print_every == 0:
                    if history['test'][-1] < best_loss:
                            best_loss = history['test'][-1]
                            best_weights = copy.deepcopy(model.state_dict())
                            patience_ = patience
                    elif patience_ == 0:
                        print("Early stopping, restoring best weights")
                        model.load_state_dict(best_weights)
                        return model, history
                    elif history['test'][-1] > best_loss or np.isnan(history['test'][-1]) or np.isinf(history['test'][-1]):
                        patience_ -= 1
                    elif float(history['test'][-1]) == 0.0:
                        patience_ -= 0.1
        
        if early_stopping and history['test'][-1] < best_loss:
            model.load_state_dict(best_weights)
            print(f"restoring weight from batch {np.argmin(history['test']) * print_every} with loss {best_loss}")
        model.eval()
        return model, history

def prep_triplet_data(MODEL=f"cardiffnlp/twitter-xlm-roberta-base-sentiment", augment=True, aug_n = 400000):
    """Prepares the data for training the triplet network
    
    Requires that the emotion dataset has been downloaded and extracted and then preprocessed to the appropriate locations.
    
    Returns:
        classes -- The list of classes
        train_triplets -- The training triplets generator
        test_triplets -- The testing triplets generator
        x_train -- The training data split 
        y_train -- The training labels split
        x_test -- The testing data split
        y_test -- The testing labels split
        """
    #emotion_df, classes = get_emotion_df()
    sentences_train, emotions_train, sentences_test, emotions_test, sentences_augs, emotions_augs, classes = get_emotion_df(MODEL, augment=augment, aug_n = aug_n)

    x_train = torch.stack([torch.tensor(sentence) for sentence in sentences_train]).to(cpu)
    x_test  = torch.stack([torch.tensor(sentence) for sentence in sentences_test ]).to(cpu)
    y_train = torch.tensor(emotions_train, dtype=torch.float32).to(cpu)
    y_test  = torch.tensor(emotions_test,  dtype=torch.float32).to(cpu)
    if augment: 
        y_aug = torch.tensor(emotions_augs, dtype=torch.float32).to(cpu)
        x_aug = torch.stack([torch.tensor(sentence) for sentence in sentences_augs]).to(cpu)

    if augment: x_train = torch.cat((x_train, x_aug), 0)
    if augment: y_train = torch.cat((y_train, y_aug), 0)

    x_train, y_train, x_test, y_test = x_train.to(cuda), y_train.to(cuda), x_test.to(cuda), y_test.to(cuda)

    train_triplets = create_triplets(x_train, y_train, batch_size=32, shuffle=True, seed=42)
    test_triplets =  create_triplets(x_test, y_test, batch_size=1, shuffle=True, seed=42)

    

    return classes, train_triplets, test_triplets, x_train, y_train, x_test, y_test

def prep_tensor_ds(x_train, y_train, x_test, y_test, bs=32):
    """Prepares the data for training the emotion classifier
    
    Arguments:
        x_train {torch.Tensor} -- The training data split
        y_train {torch.Tensor} -- The training labels split
        x_test {torch.Tensor} -- The testing data split
        y_test {torch.Tensor} -- The testing labels split
        bs {int} -- The batch size"""
    ds_train = data.TensorDataset(x_train, y_train)
    ds_train = data.DataLoader(ds_train, batch_size=bs, shuffle=True)
    ds_test = data.TensorDataset(x_test, y_test)
    ds_test = data.DataLoader(ds_test, batch_size=16, shuffle=True)
    return ds_train, ds_test
class classify_single_input(nn.Module):
    """Inherits from the siamese network and adds a head for the classification of a single input. Inherits the vector size and number of classes directly from the siamese network.
    
    Arguments:
        siamese_network {nn.Module} -- The siamese network
        
    Returns:
        nn.Module -- The emotion classifier model with the siamese network weights applied to every weight except the final layer"""
    def __init__(self, siamese_network):
        super().__init__()
        self.siamese_network = siamese_network
        self.head = nn.LazyLinear(self.siamese_network.classes)
        self.alphadropout = nn.AlphaDropout(p=0.1)

    def forward(self, x):
        inp = self.siamese_network.siamese_shared_weights(x)
        out = inp[0][:, 0, :]
        out = self.alphadropout(out)
        if self.siamese_network.head:
            out = self.siamese_network.head1(out)
        out = torch.nn.functional.softmax(out, dim=1)
        out = self.head(out)
        out = torch.nn.functional.softmax(out, dim=1)
        return out
