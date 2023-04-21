import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn as nn
import torch.utils.data as data
cuda = torch.device("cuda")
cpu = torch.device("cpu")
from tqdm.notebook import tqdm
from utils import get_emotion_df, create_triplets
import copy
import contextlib
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import get_sentence_vectors, save_file, load_file
import arrow
import torchvision.models as models

class siamese_network(nn.Module):
    """Instantiate the xlm-roberta-base-sentiment model and add a siamese network head to it to replace the sentiment classifier head with emotion classifier head. Allowes for the model to later be altered to train on non-triplet data after pre-training on triplet data.
    
    Arguments:
        classes {int} -- The number of classes to classify into
        vector_size {int} -- The size of the vector to output from the siamese network. This is the size of the vector that will be used to train the emotion classifier.
        model {str} -- The pretrained model to use. Defaults to "cardiffnlp/twitter-xlm-roberta-base-sentiment"."""
    def __init__(self, classes, vector_size=120, MODEL=f"cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        super().__init__()
        self.MODEL = MODEL
        self.config = AutoConfig.from_pretrained(MODEL, num_labels=vector_size, fine_tuning_task = 'text-classification')
        self.classes = classes
        self.vector_size = vector_size
        self.siamese_shared_weights = AutoModelForSequenceClassification.from_pretrained(self.MODEL, ignore_mismatched_sizes=True, config=self.config)

    def forward(self, x1):
        x1 = self.siamese_shared_weights.forward(x1)[0]
        return x1

def loss_fn(anchor, positive, negative, margin=1, conv_func1 = nn.Conv1d(1, 1, 5, 3, padding=1, device=cuda), conv_func2 = nn.Conv1d(1, 1, 9, 6, padding=1, device=cuda)):
    """The triplet loss function. 
    
    Arguments:
        anchor {torch.Tensor} -- The anchor vector
        positive {torch.Tensor} -- The positive vector
        negative {torch.Tensor} -- The negative vector
        margin {float} -- The margin to use for the loss function
    
    Returns:
        torch.Tensor -- The loss"""
    diff_positive_anchor = (anchor - positive)
    diff_negative_anchor = (anchor - negative)
    seperatedness_loss = torch.sum(torch.clamp(torch.sum(torch.abs(diff_positive_anchor), dim=1) - torch.sum(torch.abs(diff_negative_anchor), dim=1) + margin, min=0.0))

    # apply a conv1d to dim1 to reduce the dimensionality/add blur
    anchor_density   = conv_func1(anchor.unsqueeze(1)).squeeze(1)
    positive_density = conv_func1(positive.unsqueeze(1)).squeeze(1)
    negative_density = conv_func1(negative.unsqueeze(1)).squeeze(1)
    diff_positive_anchor = (anchor_density - positive_density) # minimize
    diff_negative_anchor = (anchor_density - negative_density) # maximize
    seperatedness_loss += torch.sum(torch.clamp(torch.sum(torch.abs(diff_positive_anchor), dim=1) - torch.sum(torch.abs(diff_negative_anchor), dim=1) + margin, min=0.0))

    anchor_density   = conv_func2(anchor.unsqueeze(1)).squeeze(1)
    positive_density = conv_func2(positive.unsqueeze(1)).squeeze(1)
    negative_density = conv_func2(negative.unsqueeze(1)).squeeze(1)
    diff_positive_anchor = (anchor_density - positive_density) # minimize
    diff_negative_anchor = (anchor_density - negative_density) # maximize
    seperatedness_loss += torch.sum(torch.clamp(torch.sum(torch.abs(diff_positive_anchor), dim=1) - torch.sum(torch.abs(diff_negative_anchor), dim=1) + margin, min=0.0))

    return seperatedness_loss

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
    siamese_model.train()

    for param in siamese_model.parameters():
        param.requires_grad = False
    for param in siamese_model.siamese_shared_weights.classifier.parameters():
        param.requires_grad = True
    for i in [11]:
        for param in siamese_model.siamese_shared_weights.roberta.encoder.layer[i].parameters():
            param.requires_grad = True

    print(f'trainable: {[name for name, param in siamese_model.named_parameters() if param.requires_grad]}')
    if early_stopping:
        best_loss = np.inf
        patience_counter = 0

    optimizer=torch.optim.Adam(siamese_model.parameters(), lr=0.001)
    if criterion is None:
        contrastive_loss = loss_fn
    else:
        contrastive_loss = criterion
    with tqdm(total=epochs * train_triplets.num_batches) as pbar:
        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(train_triplets.num_batches):
                [anchor, positive, negative], anchor_class = train_triplets.get_batch()
                optimizer.zero_grad()
                #with torch.no_grad():
                negative_ = siamese_model(negative)
                positive_ = siamese_model(positive)
                anchor_ = siamese_model(anchor)
                loss = contrastive_loss(anchor_, positive_, negative_)
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
                            negative_ = siamese_model(negative)
                            positive_ = siamese_model(positive)
                            anchor_ = siamese_model(anchor)
                            loss = contrastive_loss(anchor_, positive_, negative_)
                            valid_loss += loss.item()
                        history['test'].append(valid_loss / print_every)
                        siamese_model.train()
                        
                    history['train'].append(running_loss / print_every)
                    running_loss = 0.0
                    pbar.desc = f"train loss: {history['train'][-1]}, val loss: {history['test'][-1]}, patience {patience - patience_counter}/{patience}"
                    pbar.update(print_every)

                    if not early_stopping:
                        continue

                    if history['test'][-1] == best_loss:
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
        contrastive_loss = torch.nn.TripletMarginWithdiffanceLoss(margin=model.vector_size,reduction='mean', diffance_function=nn.Pairwisediffance(p=2.0, eps=1e-06, keepdim=False))
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

def classify_text(model, text_or_fd, MODELNAME=f"cardiffnlp/twitter-xlm-roberta-base-sentiment", bs=64, device='cuda'):
    """Classify text using the emotion classifier model
    
    Arguments:
        model {nn.Module} -- The emotion classifier model
        text {str} -- The text to classify
        MODELNAME {str} -- The name of the model to use for tokenization
        bs {int} -- The batch size
        device {str} -- The device to use for inference
    
    Returns:
        list -- The emotion predictions
        """
    model.eval
    model.to(device)
    assert isinstance(text_or_fd, (list, pd.DataFrame,pd.Series)), "text_or_fd must be a string, pandas DataFrame or pandas Series"

    # tokenize the text
    if isinstance(text_or_fd, str):
        text = [text_or_fd]
    elif isinstance(text_or_fd, pd.DataFrame):
        text = text_or_fd['text'].tolist()
    elif isinstance(text_or_fd, pd.Series):
        text = text_or_fd.tolist()
    else:
        text = text_or_fd
    sentence_vectors = get_sentence_vectors(text, MODELNAME)

    # create batches from list of sentence vectors
    batches = [sentence_vectors[i:i + bs] for i in range(0, len(sentence_vectors), bs)]

    # get predictions
    predictions = []
    for batch in tqdm(batches):
        batch = torch.stack([torch.tensor(x) for x in batch]).to(device)
        output = model(batch)
        predictions.extend(output.tolist())

    if isinstance(text_or_fd, pd.DataFrame):
        predictions = {}
        return text_or_fd
    elif isinstance(text_or_fd, pd.Series):
        return pd.Series(predictions, index=text_or_fd.index)
    else:
        return predictions
    
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
    model.siamese_network.siamese_shared_weights.classifier.requires_grad = True

    for param in model.siamese_network.siamese_shared_weights.classifier.parameters():
        param.requires_grad = True
    for i in [11]:
        for param in model.siamese_network.siamese_shared_weights.roberta.encoder.layer[i].parameters():
            param.requires_grad = True

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

def prep_triplet_data(MODEL=f"cardiffnlp/twitter-xlm-roberta-base-sentiment", augment=True, aug_n = 400000, bs=32):
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

    train_triplets = create_triplets(x_train, y_train, batch_size=bs, shuffle=True, seed=42)
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

    def forward(self, x):
        inp = self.siamese_network.forward(x)
        out = torch.nn.functional.softmax(self.head(inp), dim=1)
        return out

def classify_twitter_text(save_path, model, load_path=None, bs=100, device='cuda', drop_cols=['username', 'searchterm'], MODELNAME='cardiffnlp/twitter-xlm-roberta-base-sentiment'):
    """
    Classifies the text in a dataframe and saves the results to a file
    
    Parameters
    ----------
    save_path : str
        The path to save the file to
    model : fastai.text.learner
        The model to use to classify the text
    load_path : str, optional
        The path to load the file from, by default None
    bs : int, optional
        The batch size to use, by default 100
    device : str, optional
        The device to use, by default 'cuda'
    drop_cols : list, optional
        The columns to drop from the dataframe, by default ['username', 'searchterm']
    MODELNAME : str, optional
        The base name of the model to use (must be finetuned/altered using the pipeline), by default 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
    """
    assert os.path.exists(save_path) or load_path != None, "load_path must be a valid path to a file"  
    
    if not os.path.exists(save_path):
        df = load_file(load_path)
        df = df.dropna(subset=['text'])
        df = df.drop(columns=drop_cols)
        list_emotions = classify_text(model, df.text.tolist(), MODELNAME=MODELNAME, bs=bs, device=device)
        df['emotion'] = list_emotions
        
        save_file(df, save_path)
        return df.sort_values(by='date', inplace=False)
    else:
        return load_file(save_path).sort_values(by='date', inplace=False)

class model(nn.Module):
    """informer model for time series forecasting"""
    def __init__(self, input_size, num_classes, cash=torch.as_tensor(10000, dtype=torch.float32).to('cuda'), ticker='KR'):
        super(model, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet18(pretrained=False)
        self.portfolio_history = {'cash': [cash], 'stocks': [0], 'total': [cash], 'action': []}

        self.new_input_size = input_size
        # update resnet input size
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight = nn.Parameter(self.resnet.conv1.weight.sum(dim=1, keepdim=True))
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = nn.LazyLinear(self.num_classes)
        self.rl_head = nn.LazyLinear(1)
        self.dense_portfolio = nn.LazyLinear(5)
        self.dense_resnet = nn.LazyLinear(5)
        self.dense = nn.LazyLinear(5)

        self.cash = cash
        _ = self.action(torch.as_tensor([0], dtype=torch.float32).to('cuda'))
        self.stocks = 0
        self.stock_ticker_symbol = ticker
        self.i_train = 0
        self.i_test = 0
        self.get_stock_history(ticker)

    def forward(self, x):
        """forward pass. Here we take the resnet model and add a head for the RL agent to learn from. This forward pass will then output the action taken and the pred_price for price.
        
        """
        portfolio = torch.as_tensor([self.cash, self.stocks, self.price * self.stocks], dtype=torch.float32).unsqueeze(0).to('cuda')
        pred_price = self.resnet(x)
        
        # flatten out1 and portfolio
        
        dense_portfolio = self.dense_portfolio(portfolio)
        out1 = torch.squeeze(pred_price, 0)
        dense_portfolio = torch.squeeze(dense_portfolio, 0)

        dense_resnet = self.dense_resnet(out1)
        dense = torch.cat((dense_portfolio, dense_resnet))
        outcat2 = torch.concat((dense, out1))
        action = self.rl_head(outcat2)
        return pred_price.permute(1,0).to('cuda'), action.to('cuda')

    def action(self, amount, train=True, track=False):
        """actions are either 0, 1, or -1"""
       
        amount = amount.squeeze(0).item()
        if train: 
            price = torch.as_tensor(self.train['close'][self.i_train], dtype=torch.float32).to('cuda')
            self.i_train += 1
            if self.i_train >= len(self.train): self.i_train = 0
        else: 
            price = torch.as_tensor(self.test['close'][self.i_test], dtype=torch.float32).to('cuda')
            self.i_test += 1
            if self.i_test >= len(self.test): self.i_test = 0

        self.price = price
        price.requires_grad = True
        old_value = self.get_value(price)

        if amount > 0.33:
            amount = 1
        elif amount < -0.33:
            amount = -1
        else:
            amount = 0

        self.cash -= price * amount
        self.stocks += amount

        new_value = self.get_value(price)

        # maximize profit (reward)
        change = new_value - old_value
        change = -change
        
        if track:
            self.portfolio_history['cash'].append(self.cash)
            self.portfolio_history['stocks'].append(self.stocks)
            self.portfolio_history['total'].append(self.cash + self.stocks * price)
            self.portfolio_history['action'].append(['sell', 'hold', 'buy'][amount + 1])
        
        loss = torch.as_tensor(change, dtype=torch.float32).to('cuda')
        loss = torch.functional.F.softmax(loss, dim=0)
        return loss


    def get_value(self, price):
        return self.cash + self.stocks * price

    def get_stock_history(self, ticker):
        self.train = load_file(f'data/Stock/{ticker}.parquet')
        self.test = load_file(f'data/Stock/{ticker}.parquet')
        self.train['date'] = self.train['date'].apply(lambda x: arrow.get(str(x)[:10]).format('YYYY-MM-DD') if len(str(x)) > 7 else None).dropna()
        self.test['date'] = self.test['date'].apply(lambda x: arrow.get(str(x)[:10]).format('YYYY-MM-DD') if len(str(x)) > 7 else None).dropna()

def emotion_classifier_load(MODEL='cardiffnlp/twitter-xlm-roberta-base-sentiment', vector_size=120, max_epocks=10, early_stopping=True, patience=500, print_every=500, device='cuda') -> nn.Module:
    """Loads the emotion classifier model and trains it if it does not exist
    
    Keyword Arguments:
        MODEL {str} -- The model to use (default: {'cardiffnlp/twitter-xlm-roberta-base-sentiment'})
        vector_size {int} -- The vector size of the siamese network (default: {120})
        classes_len {int} -- The number of classes (default: {classes_len})
        max_epocks {int} -- The maximum number of epocks to train for (default: {10})
        classes {list} -- The list of classes (default: {classes})
        early_stopping {bool} -- Whether to use early stopping (default: {True})
        patience {int} -- The patience for early stopping (default: {500})
        print_every {int} -- How often to print the loss (default: {500})
        device {str} -- The device to use (default: {'cuda'})
        
        Returns:
            nn.Module -- The emotion classifier model
        """
    
    if not os.path.exists('models/emotion_classifier.pt') or not os.path.exists('models/siamese_model.pt'):
        classes, train_triplets, test_triplets, x_train, y_train, x_test, y_test = prep_triplet_data(MODEL=MODEL, augment=True, aug_n = 400000)
        ds_train, ds_test = prep_tensor_ds( x_train, y_train, x_test, y_test)
        classes_len = len(classes)
    else:
        classes_len = 29

    if not os.path.exists('models/siamese_model.pt'):
        siamese_network_model = siamese_network(classes_len, vector_size=vector_size, MODEL=MODEL).to(device)
        siamese_model, history = pre_train_using_siamese(train_triplets, test_triplets, siamese_network_model, epochs=max_epocks, classes=classes, early_stopping=early_stopping, patience=patience, print_every=print_every)
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(siamese_model.state_dict(), f'models/siamese_model.pt')

        # plot the train and test loss over time on the same plot
        fig = plt.figure(figsize=(10, 5))
        plt.plot(history['train'], label='pretraining train loss')
        plt.plot(history['test'], label='pretraining test loss')
        plt.legend()
        plt.show()

        # save plot of train and test loss over time along with history
        if not os.path.exists('results'):
            os.makedirs('results')
        fig.savefig('results/pretrain_emotion_history.jpg')
        save_file(history, 'results/pretrain_emotion_history.parquet')
        plt.close(fig)
    else:
        siamese_network_model = siamese_network(classes_len).to(device)
        siamese_network_model.load_state_dict(torch.load(f'models/siamese_model.pt'))

    """In this step, we finally train the last two weight layers to convert the siamese network into a classifier. Later, we can consider unfreezing a few of the earlier layers to improve performance with a lower learning rate."""

    if not os.path.exists('models/emotion_classifier.pt'):
        if not os.path.exists('models/siamese_model.pt'):
            siamese_network_model = siamese_network(classes_len, vector_size=vector_size, MODEL=MODEL).to(device)
        model = classify_single_input(siamese_network_model)
        model = model.to(device)
        model, history = train_emotion_classifier(model, ds_train, ds_test, epochs=max_epocks, early_stopping=early_stopping, patience=patience, print_every=print_every)

        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), f'models/emotion_classifier.pt')

        # plot the train and test loss over time on the same plot
        fig = plt.figure(figsize=(10, 5))
        plt.plot(history['train'], label='train loss')
        plt.plot(history['test'], label='test loss')
        plt.legend()
        plt.show()

        # save plot of train and test loss over time along with history
        if not os.path.exists('results'):
            os.makedirs('results')
        fig.savefig('results/emotion_history.jpg')
        save_file(history, 'results/emotion_history.parquet')
        plt.close(fig)
    else:
        model = classify_single_input(siamese_network_model)
        model = model.to(device)
        model.load_state_dict(torch.load(f'models/emotion_classifier.pt'))

    return model