from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64, os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from compress_pickle import dump, load
from yahooquery import Ticker
import timeit
import datetime
import re
from getpass import getpass
from shutil import rmtree
import os
pd.set_option('io.parquet.engine', 'pyarrow')
import torch
import subprocess
import urllib.request
import shutil
import os
from transformers import AutoTokenizer
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from nlpaug.util import Action
from tqdm import tqdm
import random
from flair.data import Sentence
from flair.nn import Classifier
from gc import collect
import contextlib
import io
import zipfile
from sklearn.model_selection import train_test_split
import arrow
import snscrape.modules.twitter as sntw
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
cuda = torch.device("cuda")
cpu = torch.device("cpu")
from time import sleep
import random
import sys

def download_dataset(url:str, unzip:bool=True, delete_zip:bool=True, files_to_move:dict = {}, delete=False, dest_name:str = None, verbose:bool = True, min_files_expected=1):
    """Downloads the datasets from kaggle using the official kaggle api.
    
    See this forumn for more information, as the official documentation is lacking:
        https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python
        
    Args:
        url (str): The url of the dataset. Can be found on the kaggle website.
        unzip (bool): Whether to unzip the dataset or not.
        delete_zip (bool): Whether to delete the zip file after unzipping.
        files_to_move (dict): A dictionary of files to keep. This is used to determine whether the dataset has already been downloaded.
        delete (bool): Whether to delete the dataset folder after moving the files.
        dest_name (str): The name of the folder to move the files to.
        verbose (bool): Whether to print the progress of the download.
        min_files_expected (int): The minimum number of files expected in the folder. If the number of files is less than this, the dataset will be downloaded. A quick and lazy way to check if the dataset has been downloaded. Open to imporvement."""
    from kaggle.api.kaggle_api_extended import KaggleApi 

    api = KaggleApi()
    api.authenticate()

    if "datasets" in url: url = url.split("datasets/")[-1]
    if not os.path.exists('data/'): os.mkdir('data/')
    if not os.path.exists('data/{}'.format(dest_name)): os.mkdir('data/{}'.format(dest_name))
    if delete_zip:
        for file in glob('data/*.zip'):
            os.remove(file)

    aquire_files_needed = True if len(glob(f'data/{dest_name}/*.parquet')) < min_files_expected else False
    
    # if any requested files are not in the folder, download the dataset
    if aquire_files_needed:
        api.dataset_download_files(url, path='data/', unzip=unzip, quiet=not verbose)

        for k, v in files_to_move.items():
            if os.path.exists(f'data/{k}'):
                if not os.path.exists(f'data/{v}'):
                    os.rename(f'data/{k}', f'data/{v}')

        if delete_zip:
            for file in glob('data/*.zip'):
                os.remove(file)

        if delete:
            folder = url.split('/')[-1]
            if os.path.exists(f'data/{folder}'):
                os.rmdir(f'data/{folder}')
    else:
        if verbose: print(f"Dataset ({dest_name}) already downloaded.")

    # the above function does not remove/transfer files from the folder 'data/Data' correctly, so we need to do it manually
    if os.path.exists('data\Data\Stocks'):
        os.removedirs('data/Stock')
        os.rename('data/Data/Stocks', 'data/Stock')
        rmtree('data/Data')

def convert_project_files_to_parquet():
    """"convert the files to parquet format, which is a much better for this project. Will convert all csv, xlsx, txt, json, dat and pkl files in data/ to parquet and then delete the original files (to save space).
    
    Removes the following columns from the dataframes:
        - Dividends
        - Stock Splits
        
    Converts the following columns to the following format:
        - Date: YYYY-MM-DD
            
    Returns:
        None
                """
    for file_type in ['csv', 'xlsx', 'txt', 'json', 'dat', 'pkl']:
        files_to_compress = glob(f'data/*/*.{file_type}') + glob(f'data/*/*/*.{file_type}') + glob(f'data/*.{file_type}')
        for file in files_to_compress:
            if not os.path.exists(file.replace(f'.{file_type}', '.parquet')):
                df = load_file(file)
                df.columns = [x.lower() for x in df.columns]
                if 'dividends' in df.columns or 'stock splits' in df.columns:
                    df = df.drop(columns=['dividends', 'stock splits'], inplace=False, errors='ignore')
                if 'date' in df.columns:
                    df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
                if type(df) != bool: df.to_parquet(file.replace(f'.{file_type}', '.parquet'), compression='brotli', engine='pyarrow')
            if os.path.exists(file):
                os.remove(file)
    # remove npy files
    for file in glob('data/*.npy'):
        os.remove(file)

def load_file(file:str):
    """Reads file into a pandas dataframe. Supports csv, xlsx, txt, json, lz4, dat and pkl files.

    Very opinionated. Will lowercase all column names, and will convert the date column to a string in the format YYYY-MM-DD. This makes it so there is less data processing each time a new dataset gets created.
    
    args:
        file path (str): The file to load."""
    file_type = file.split('.')[-1]
    if os.path.exists(file):
        if file_type == 'parquet':
            df = pd.read_parquet(file)
            df.columns = [x.lower() for x in df.columns]
            return df
        elif file_type == 'csv':
            df =  pd.read_csv(file, low_memory=False, parse_dates=True, infer_datetime_format=True, on_bad_lines='skip', encoding_errors= 'replace')
            df.columns = [x.lower() for x in df.columns]
            if 'date' in df.columns:
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
            return df
        elif file_type == 'xlsx':
            df =  pd.read_excel(file, parse_dates=True)
            df.columns = [x.lower() for x in df.columns]
            if 'date' in df.columns:
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
            return df
        elif file_type == 'pkl':
            df =  load(file)
            df.columns = [x.lower() for x in df.columns]
            if 'date' in df.columns:
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
            return df
        elif file_type in ['.lz4', 'lz4']:
            df =  load(file, compression='lz4')
            df.columns = [x.lower() for x in df.columns]
            if 'date' in df.columns:
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
            return df
        elif file_type == 'json':
            df =  pd.read_json(file)
            df.columns = [x.lower() for x in df.columns]
            if 'date' in df.columns:
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
            return df
        elif file_type == 'txt':
            df =  pd.read_csv(file, sep='\t', parse_dates=True, infer_datetime_format=True, on_bad_lines='skip', encoding_errors= 'replace')
            df.columns = [x.lower() for x in df.columns]
            if 'date' in df.columns:
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
            return df
        elif file_type == 'dat':
            df =  pd.read_csv(file, sep='\t', low_memory=False, parse_dates=True, infer_datetime_format=True, on_bad_lines='skip', encoding_errors= 'replace')
            df.columns = [x.lower() for x in df.columns]
            if 'date' in df.columns:
                df = df.dropna(subset=['date'])
                df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
            return df
        else:
            try:
                return load(file)
            except:
                assert False, f'File type {file_type} not supported.'
    else:
        print(f'File {file} does not exist.')

def save_file(df, file:str, force_type=None):
    """Saves a pandas dataframe to a file. Supports parquet, pkl, csv and xlsx files.

    Very opinionated about column names. All columns will be lower case. If a column is named 'date', it will be converted to a string in the format 'YYYY-MM-DD'.
    
    args:
        df (pd.DataFrame): The dataframe to save.
        file path (str) + extension: The file to save to.
        columns (list): The columns to save. If None, all columns will be saved with default names."""
    file_type = file.split('.')[-1]
    #print(type(df[0]), type(df), df)
    if type(df) == list:
        df ={'0': df}
        if force_type:
            df = pd.DataFrame(data=df, dtype=force_type)
        else:
            df = pd.DataFrame(data=df)
    if type(df) == dict:
        df = pd.DataFrame(data=df)
    if file_type == 'parquet':
        if force_type:
            df = df.astype(force_type)
        df.columns = [x.lower() for x in df.columns]
        if 'date' in df.columns:
            df['date'] = df['date'].apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))
        df.to_parquet(file, compression='brotli', engine='pyarrow')
    elif file_type == 'pkl':
        df.columns = [x.lower() for x in df.columns]
        dump(df, file + '.lz4', compression='lz4')
    elif file_type == 'csv':
        df.columns = [x.lower() for x in df.columns]
        df.to_csv(file, index=False)
    elif file_type == 'xlsx':
        df.columns = [x.lower() for x in df.columns]
        df.to_excel(file, index=False)

def fernet_key_encryption(password:str, name:str):
    """Encrypts and decrypts a key using Fernet encryption. 

    If you need to change the keys or password, delete the relevent .secret keys file and run this section again.

    salt.secret is a non-sensitive file that is used to both generate the encryption key as well as decryption. If this key is lost, the encrypted files are lost and you will need to re-enter the api keys. It should automatically be generated if it is not found.
    
    Args:
    
        password (str): The password to encrypt the key with.
        
        name (str): The name of the key to encrypt.
        
    Returns:
    Saves the encrypted key to a file, creates a salt file if needed, and returns the decrypted key."""
    # Convert to type bytes
    password = password.encode()
    
    # generate salt
    if not os.path.exists('salt.secret'):
        #search for .secret files and delete them if the required salt file is not found
        for file in glob('*.secret'):
            os.remove(file)

        salt = os.urandom(16)
        with open('salt.secret', 'wb') as f:
            f.write(salt)
    else:
        with open('salt.secret', 'rb') as f:
            salt = f.read()

    # derive key from password
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA512(),
        length=32,
        salt=salt,
        iterations=1000000) # lower this if it takes too long

    key = base64.urlsafe_b64encode(kdf.derive(password))
    fernet = Fernet(key)

    if os.path.exists(f'{name}_token.secret'):
        with open(f'{name}_token.secret', 'rb') as f:
            token = f.read()
        input_message = fernet.decrypt(token)
        return input_message.decode()

    else:
        input_message = getpass(f'Please Enter your {name} API key. Seperate any username(first)/passwords(second) with a space: ').encode()
        token = fernet.encrypt(input_message)
        del input_message
        gc.collect()
        with open(f'{name}_token.secret', 'wb') as f:
            f.write(token)
        return token.decode()

def augment_list_till_length(l, maxlength=400000, num_thread = 12, mapping = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}):
    """Augments a list of sentences until it reaches a certain length.
    
    args:
        l (list): The list of sentences to augment.
        maxlength (int): The maximum length of the augmented list.
        num_thread (int): The number of threads to use.
        mapping (dict): The mapping of the sentiment labels to numerical values.
            
    returns:
        a classifed list of sentences with the same emotional valence as the original sentences.

    Logic:
        1. Add bert generated similar sentences to the list.
        2. Add synonym generated similar sentences to the list.
        3. verify that the augmented sentences have the same emotional valence as the original sentences.
        4 add typos to the list.
        5. verify that the augmented sentences have the same emotional valence as the original sentences.
        6. repeat steps 1-5 until the list is at least maxlength long.        
    """
    
    tagger = Classifier.load('sentiment')
    
    def sentiment_(l):
        """Ensures that, at a minimum, the augmented sentences have the same emotional valence as the original sentences.
        
        args:
            l (list): The list of sentences to augment.
            
        returns:
            a classifed list of sentences with the same emotional valence as the original sentences.
        """
        sentences = [Sentence(s) for s in l]
        tagger.predict(sentences)
        r = range(len(sentences))
        results = []
        for i in r:
            s = sentences[i].get_labels()[0].value
            results.append(s)
        sentences = [mapping[r] for r in results]
        return sentences
    
    aug_pre = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", aug_min=1, aug_max=3, device='cuda')
    aug_pre2 = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=1)
    aug = naf.Sometimes([
        nac.RandomCharAug(action="insert", aug_char_min=0, aug_char_max=2, aug_word_min=0, aug_word_max=2, aug_word_p=0.05, aug_char_p=0.05),
        nac.KeyboardAug(aug_char_p=0.05, aug_word_p=0.3, aug_word_min=0, aug_word_max=2, aug_char_min=0, aug_char_max=2),
        naw.SpellingAug(aug_min=0, aug_max=3, aug_p=0.05),
        ])

    print(f'step 1: augmenting len(list) {len(l)} to {maxlength//7} using naw.ContextualWordEmbsAug')
    with tqdm(total=maxlength, dynamic_ncols=True, initial=len(l)) as pbar:
        l_frozen_size = len(l)
        augments = set(l)
        sentences = sentiment_(l)
        if len(l) < maxlength//2:

            while len(augments) < maxlength//7:
                n = max(1, int(round(maxlength//6//len(l),0)))
                for i in range(l_frozen_size):
                    potential1 = aug_pre.augment(l[i], n=n, num_thread=min(num_thread, n))
                    pot_senti1 = sentiment_(potential1)
                    pot_senti1 = [potential1[j] for j in range(len(potential1)) if pot_senti1[j] == sentences[i]]
                    augments.update(pot_senti1)
                    pbar.update(len(augments) - pbar.n)
                    if len(augments) >= maxlength//7:
                        break

            l = [x for x in augments]
            augments = set(l)
            sentences = sentiment_(l)
            l_frozen_size = len(l)
            while len(augments) < maxlength//2:
                n = max(1, int(round(maxlength//2//len(l),0)))
                for i in range(l_frozen_size):
                    potential2 = aug_pre2.augment(l[i], n, num_thread=1)
                    pot_senti2 = sentiment_(potential2)
                    pot_senti2 = [potential2[j] for j in range(len(potential2)) if pot_senti2[j] == sentences[i]]
                    augments.update(pot_senti2)
                    pbar.update(len(augments) - pbar.n)
                    if len(augments) >= maxlength//2:
                        break
            l = [x for x in augments]
            
        print(f'step 2: augmenting len(list) {len(l)} to {maxlength} using synonyms and spelling errors')
        augments = set(l)
        pbar_start = pbar.n
        n = max(1, int(round(maxlength//len(l),0)))
        while len(augments) < maxlength:
            for i in range(len(l)):
                augments.update(aug.augment(l[i], n=n, num_thread=num_thread))
                pbar.update(len(augments) - pbar.n + pbar_start)
                if len(augments) >= maxlength:
                    break
    

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.suppress(Exception):
            augments = [x for x in augments]
            augments = random.sample(augments, len(augments))
            del aug, aug_pre, aug_pre2, augs 
        collect()
    return augments[:maxlength]

def parse_emotion_dataframes(selection: int = [0, 1, 2, 3, 4, 5], ensure_only_one_label: bool = True, augment_data=True, aug_n = 400000):
    """Parses the emotion dataframes. Normalizes each of the specific 29 dataset labels to the same label names, then combines the five of them into one dataframe. The resulting dataset is highly unbalanced, with predominant negative emotions.
    
    Args: 
        selection (int): The emotion dataframes to parse. 0 = https://www.kaggle.com/datasets/mathurinache/goemotions, [1, 2, 3] = www.kaggle.com/datasets/parulpandey/emotion-dataset, 4 = https://www.kaggle.com/datasets/kosweet/cleaned-emotion-extraction-dataset-from-twitter
        ensure_only_one_label (bool): If True, ensures that each row only has one label. If False, allows multiple labels per row.

    Returns:
        df (pd.DataFrame): The combined dataframe of the selected datasets.
    """
    df0, df1, df2, df3, df4, df5 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    def df_load_(file, drop_cols, new_cols, rename_cols=[]):
        """Loads the dataframe and drops the columns that are not needed."""
        df = load_file(file)
        if 'label' in df.columns:
            df = pd.get_dummies(df, columns=['label'])
            df.columns = [x.replace('label_', '') for x in df.columns]
        if 'Emotion' in df.columns:
            df = pd.get_dummies(df, columns=['Emotion'])
            df.columns = [x.replace('Emotion_', '') for x in df.columns]
        df = df.drop(columns=drop_cols)
        if rename_cols.__len__() > 0:
            df.columns = rename_cols
        for col in new_cols:
            if col not in df.columns:
                df[col] = 0
        all_emotions = list(set([x for x in df.columns[df.columns != 'text'] if x not in drop_cols]))
        df = df.fillna(0)
        # Remove the numerical rows that add up to greater than 1 or equal to 0 and reset the index
        if ensure_only_one_label:
            e = df[all_emotions]
            df = df[(e.sum(axis=1) <= 1) & (e.sum(axis=1) > 0)].reset_index(drop=True)

        return df, all_emotions

    if 0 in selection:
        df0, all_emotions = df_load_('data/Emotions/goemotions.parquet', ['example_very_unclear', 'rater_id', 'created_utc', 'parent_id', 'subreddit', 'author', 'id', 'link_id'], ['happiness'])

    if 1 in selection:
        df1,all_emotions = df_load_('data/Emotions/training.parquet', [], all_emotions, ['text', 'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])

    if 2 in selection:
        df2, all_emotions = df_load_('data/Emotions/validation.parquet', [], all_emotions, ['text', 'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])

    if 3 in selection:
        df3,all_emotions = df_load_('data/Emotions/test.parquet', [], all_emotions, ['text', 'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])
    
    if 4 in selection:
        df4,all_emotions = df_load_('data/Emotions/dataset(clean).parquet', ['Original Content'], all_emotions, ['text','anger', 'disappointment', 'happiness'])

    if 5 in selection:
        # 0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'
        df5, all_emotions = df_load_('data/Emotions/dailydialog.parquet', [], all_emotions, [])


    df = pd.concat([df0, df1, df2, df3, df4, df5], ignore_index=True)
    df = df.drop_duplicates(subset=['text'])
    df.fillna(0, inplace=True)
    df = df.reset_index(drop=True)
    df_augs = pd.DataFrame(columns=df.columns)


    x_train, x_test, y_train, y_test = train_test_split(df.text.values, df[all_emotions].values, test_size=0.2, random_state=42, stratify=df[all_emotions].values)
    df_train = pd.DataFrame({'text': x_train})
    df_test = pd.DataFrame({'text': x_test})
    for i in range(len(all_emotions)):
        df_train[all_emotions[i]] = y_train[:, i]
        df_test[all_emotions[i]] = y_test[:, i]

    if augment_data:
        subset_lists_per_label = {df_train.columns[1:].tolist()[i]: df_train[df_train.columns[1:].tolist()[i]] == 1 for i in range(len(df_train.columns[1:].tolist()))}
        for label in subset_lists_per_label:
            subset_to_add = df_train[subset_lists_per_label[label]].text.values.tolist()

            subset_to_add = augment_list_till_length(subset_to_add, num_thread=12, maxlength=aug_n)
            subset_labels = [1]* len(subset_to_add)

            subset = pd.DataFrame({'text': subset_to_add, label: subset_labels})
            df_augs = pd.concat([df_augs, subset], ignore_index=True)

            df_augs = df_augs[~df_augs.text.isin(df.text.values.tolist())]
            df_augs.fillna(0, inplace=True)
            df_augs.drop_duplicates(subset=['text'], inplace=True)
            df_augs = df_augs.reset_index(drop=True)

    return df_train, df_test, df_augs

def df_to_datetime(df: pd.DataFrame, offset_days: int = 0):
    """Converts the date column to a datetime index and drops the date column"""
    if not isinstance(df.index, pd.DatetimeIndex):
        #drop the time of day
        #df.date = df.date.dt.date
        if 'date' in df.columns: col = 'date'
        elif 'Date' in df.columns: col = 'Date'
        else: 
            for collumn in df.columns:
                if 'date' in col.lower():
                    col = collumn
                    break
        # df[col] = df[col].apply(lambda x: x.split(' ')[0])
        # offset_days = pd.Timedelta(f'{offset_days} days') #pd.Timedelta(days=offset_days)
        # time = pd.to_datetime(df[col], format=r"%Y-%m-%d")
        # time = time.apply(lambda x: x + offset_days)
        # # convert to days
        # df.index = time

        # print(time)
        # df.drop(columns=[col], inplace=True)
        df[col] = df[col].apply(lambda x: arrow.get(x))
        if offset_days != 0: df[col] = df[col].apply(lambda x: x.shift(days=offset_days))
        df[col] = df[col].apply(lambda x: x.format('YYYY-MM-DD'))
        df[col] = pd.to_datetime(df[col], format=r"%Y-%m-%d")
        df.index = df[col]
    return df

def interpolate_months_to_days(df: pd.DataFrame, extend_trend_to_today: bool = False, offset_days: int = 0):
    """Interpolates the dataframe to account for missing days. E.g, Macro data is usually available on a monthly basis.
    
    Args: DataFrame, extend_trend_to_today (bool): If True, the index will be extended to today (only enabled for monthly data).)"""
    #check if index is already datetime
    df = df_to_datetime(df)

    df = df_to_datetime(df, offset_days=offset_days)

    #check if the indexonthly
    if abs(df.index[-2:-1] - df.index[-1:]) > abs(14* pd.Timedelta('1 day')):
        monthly = True
    else:
        monthly = False

    if extend_trend_to_today and monthly:
        print('Extending trend to today. Warning: This may not be accurate and is only suggested for monthly economic data.')
        days_to_extend = (pd.to_datetime('today') - df.index[-1] + pd.Timedelta('1 day')).days
        
        cols = [[] for x in range(len(df.columns))]
        for col in range(len(df.columns)):
            fit = np.polyfit(range(len(df.iloc[-7:])), df.iloc[-7:, col], 1)
            # this next step needs to account for the change in the index from months to days
            cols[col] = np.poly1d(fit)(range(len(df.iloc[-7:]), len(df.iloc[-7:]) + days_to_extend)) 

        # for every column in the cols, create a new column in a new dataframe
        df2 = pd.DataFrame(cols).T
        df2.columns = df.columns # copy over the column names
        df2.index = pd.date_range(df.index[-1] + pd.Timedelta('1 day'), periods=days_to_extend, freq='D') # create a new index without any overlap
        df = pd.concat([df, df2]) # concat the two dataframes

        
    df = df.resample('D').interpolate(method='linear') # interpolate the missing days
    
    return df

def intersect_df(df1: pd.DataFrame, df2: pd.DataFrame, interpolate_to_days: bool = False, extend_trend_to_today: bool = False, offsett=0):
    """Performantly Intersects two dataframes based on their (datetime) index.
    
    Args: 
        - Two dataframes with a datetime index.
        - interpolate_to_days (bool): If True, the dataframes will be interpolated to account for missing days. E.g, Macro data is usually available on a monthly basis.
        - extend_trend_to_today (bool): If True, the index will be extended to today (only enabled for monthly data).
        - offset_2nd_df_by_days (int): The number of days to offset the second dataframe by n. This is useful for next day prediction.
    
    Returns: The datapoints shared between the two datasets."""



    if interpolate_to_days:
        df1 = interpolate_months_to_days(df1, extend_trend_to_today, offset_days=offsett)
        df2 = interpolate_months_to_days(df2, extend_trend_to_today, offset_days=offsett)

    df1 = df1[df1.index.isin(df2.index)]
    df2 = df2[df2.index.isin(df1.index)]

    return df1, df2
class get_macroeconomic_data ():
    """Aquire historical macroeconomic data from different sources.
    
    Not yet implemented:
    """
    def __init__(self, path):
        raise NotImplementedError

class aquire_stock_search_terms():
    """
    Gather the company info for all the ticker symbols and return a dataframe with relevant search terms for each company.
    If the stocks dataset is updated on kaggle, compank_list.pkl needs to be deleted and this run again if the symbols have changed. 

        args:

            - file_path (str): The path to the stock data.
            - file_ext (str): The file extension of the stock data.

        functions:
            >>> verify_dataset_downloaded: Check if the stock data is downloaded.
            >>> load_symbols: Load all the stock symbols.
            >>> ticker_list_to_dataframe: Convert the list of tickers to a dataframe.
            >>> get_company_list: Get the company list from yahoo finance.    
            >>> process_information: Process the information from yahoo finance.
            >>> get_quote_type: Return the quote type for a given ticker.
            >>> get_sector: Return the sector for a given ticker.
            >>> get_industry: Return the industry for a given ticker.
            >>> get company_officers: Return the company officers for a given ticker.

        All the functions are called when the class is instantiated.

        Returns:
            - A dataframe with relevant search terms for each company in the stock dataset.
            - Results are saved to file to avoid having to run this again.

    """
    def __init__(self, file_path = 'data/Stock/', file_ext = '.parquet'):
        """Aquire the company info for all the ticker symbols and return a dataframe with relevant search terms for each company."""
        self.file_path = file_path
        self.file_ext = file_ext

        if self.verify_dataset_downloaded():
            self.load_symbols()
            self.ticker_list_to_dataframe()
            self.get_company_list()
            self.process_information()

    def load_symbols(self):
        """Load all the stock symbols."""
        self.stocks_symbols = load_file("data/Stock_List.parquet")['symbol'].tolist()
        return self.stocks_symbols

    def ticker_list_to_dataframe(self):
        """Convert the list of tickers to a dataframe."""
        self.data = pd.DataFrame(self.stocks_symbols, columns = ['ticker'])

    def get_quote_type(self, index):
        """Return the quote type for a given ticker."""
        if type(index) == int:
            if 'quoteType' in self.yh_tickers[index]:
                return self.yh_tickers[index]['quoteType']
        return ""

    def get_sector(self, index):
        """Return the sector for a given ticker."""
        if type(index) == int:
            if 'sector' in self.yh_tickers[index]:
                return self.yh_tickers[index]['sector']
        return ""

    def get_company(self, index):
        """Return the company name for a given ticker."""
        if type(index) == int:
            if 'longName' in self.yh_tickers[index]:
                return self.yh_tickers[index]['longName']
            if 'longBusinessSummary' in self.yh_tickers[index]:
                name =  self.yh_tickers[index]['longBusinessSummary'].replace(". ", " ").replace(",", "")
                # if name contains break_words, split on them and take the first word
                
                # words to break on but also keep in the name
                break_words = [' Inc ', ' Corp ', ' Ltd ', ' LLC ', ' LTD ', ' Corporation ', ' Company ', ' Group ', ' Holdings ', ' Systems ', ' Technologies ', ' Technology ', ' Services ', ' Solutions ', " Bancorp ", " Limited ", " develops ", " researches ", "S.A", " Inc.", " N.V", " Co ", " Corp. "]
                for word in break_words:
                    if word in name:
                        name = name.split(word)[0] + word
                        #remove trailing spaces
                        name = name.strip()
                
                # exclusive words to break on
                break_words2 = [" is ", " and sells ", " does ", " engages ", " operates ", " provides ", " de ", " focuses ", " supports ", " - ", " distributes ", " specializes ", " (", " originates ", " owns ", " acquires ", " invests ", " offers ", " through ", " together ", " a ", " an ", " supplies ", "explores ", " designs ", " formerly " , " manufactures "," holds ", " included ", " ranks "]
                for word in break_words2:
                    if word in name:
                        name = name.split(word)[0]
                
                # words to break and remove from the front of the name
                break_words_front = ["2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030", " VA "]
                for word in break_words_front:
                    if word in name:
                        name = name.split(word)[-1]
                        name = name.strip()
                
                if name[-1] == ".":
                    name = name[:-1]

                return name
            if 'shortName' in self.yh_tickers[index]:
                return self.yh_tickers[index]['shortName']
        return ""

    def get_company_officers(self, index):
        """Return the company officers for a given ticker."""
        if type(index) == int:
            if 'companyOfficers' in self.yh_tickers[index]:
                if len(self.yh_tickers[index]["companyOfficers"]) > 0:
                    if 'totalPay' in  self.yh_tickers[index]["companyOfficers"]:
                        officers = self.yh_tickers[index]["companyOfficers"]
                        officers = sorted(officers, key = lambda x: x['unexercisedValue'] + x['exercisedValue'] + x['totalPay'], reverse = True)
                    elif 'exercisedValue' in  self.yh_tickers[index]["companyOfficers"]:
                        officers = self.yh_tickers[index]["companyOfficers"]
                        officers = sorted(officers, key = lambda x: x['unexercisedValue'] + x['exercisedValue'], reverse = True)
                    else:
                        officers = self.yh_tickers[index]["companyOfficers"]
                        officers = sorted(officers, key = lambda x: x['unexercisedValue'], reverse = True)

                    officer = officers[0]['name'].replace("Mr. ", "").replace("Ms. ", "").replace("Dr. ", "").replace("Mrs. ", "").replace(".", "").split(" ")
                    officer = [x for x in officer if len(x) > 1]
                    officer = " ".join(officer)
                    return officer
        return ""

    def get_industry(self, index):
        """Return the industry for a given ticker."""
        if type(index) == int:
            if 'industry' in self.yh_tickers[index]:
                return self.yh_tickers[index]['industry']
        return ""

    def get_company_list(self):
        """Return a list of company information for later processing. Caches results to file."""
        if not os.path.exists('company_list.pkl'):
            self.yh_tickers_init = [Ticker(ticker) for ticker in self.stocks_symbols]
            self.yh_tickers = [{} for _ in self.stocks_symbols]
            length = len(self.yh_tickers_init)
            errors = 0
            recovered_errors = 0
            start = timeit.default_timer()
            for t in range(len(self.yh_tickers_init)):
                try:
                    sym = self.stocks_symbols[t]
                    self.yh_tickers[t] = self.yh_tickers_init[t].asset_profile[sym]
                    if type(self.yh_tickers[t]) != dict:
                        self.yh_tickers[t] = {}
                        errors+=1
                except Exception as e:
                    self.yh_tickers[t] = {}
                    errors+=1
                    print(e)
                if 'longBusinessSummary' in self.yh_tickers[t] or 'companyOfficers' in self.yh_tickers[t] or 'industry' in self.yh_tickers[t]:
                    more_info = self.yh_tickers_init[t].quote_type[sym]
                    if type(more_info) != dict:
                        more_info = {}
                    self.yh_tickers[t] = {**self.yh_tickers[t], **more_info}
                else:
                    try:
                        if len(self.yh_tickers[t]) == 0:
                            self.yh_tickers[t] = self.yh_tickers_init[t].quote_type[sym]
                        else:
                            more_info = self.yh_tickers_init[t].quote_type[sym]
                            if type(more_info) != dict:
                                more_info = {}
                            self.yh_tickers[t] = {**more_info, **self.yh_tickers[t]}
                        recovered_errors += 1
                        
                    except Exception as e:
                        print(e)

                #time.sleep(0.2)
                time_now = timeit.default_timer()

                time_left = round((time_now - start)/(t/length + 1e-10) - (time_now - start), 0)
                time_left = str(datetime.timedelta(seconds=time_left))

                print(f"{round(100*t/len(self.yh_tickers),1)}% complete.", "Estimated time left:", time_left, "seconds. Symbol:", self.stocks_symbols[t], type(self.yh_tickers[t]), "--- errors:",errors, f"of current: {t}, total: {len(self.yh_tickers)}.", "Recovered_errors:", recovered_errors,"          ",end = '\r')
            with open('company_list.pkl', 'wb') as f:
                dump(self.yh_tickers, f, compress='lz4')
        else:
            with open('company_list.pkl', 'rb') as f:
                self.yh_tickers = load(f, decompress='lz4')
        self.missed = []
        for name in self.yh_tickers:
            if type(name) == str:
                self.missed.append(name)
    
    def get_year_first_traded (self, index):
        """Return the year the company was founded. If not found, return an empty string. Has a chance of returning the wrong year."""
        if type(index) == int:
            if 'firstTradeDateEpochUtc' in self.yh_tickers[index]:
                return self.yh_tickers[index]['firstTradeDateEpochUtc']
        return ""

    def get_year_founded(self, index):
        """Return the year the company was founded. If not found, return an empty string. Has a chance of returning the wrong year."""
        if type(index) == int:
            if "longBusinessSummary" in self.yh_tickers[index]:
                # check if "in ^d {4}$"
                summary = self.yh_tickers[index]["longBusinessSummary"]
                # find 4 digit sequences and return the smallest one
                years = re.findall(r'\d{4}', summary)
                current_year = datetime.datetime.now().year
                years = [int(y) for y in years if int(y) > 1750 and int(y) < current_year]
                if len(years) > 0:
                    return min(years)
        return ""

    def process_information(self):
        """Process the information from the company list."""
        self.data.columns = [x.lower() for x in self.data.columns]
        self.data['company'] = [self.get_company(t) for t in range(len(self.yh_tickers))]
        self.data["top_executive"] = [self.get_company_officers(t) for t in range(len(self.yh_tickers))] 
        self.data['industry'] = [self.get_industry(t) for t in range(len(self.yh_tickers))]
        self.data['quotetype'] = [self.get_quote_type(t) for t in range(len(self.yh_tickers))]
        self.data['sector'] = [self.get_sector(t) for t in range(len(self.yh_tickers))]
        self.data['first_traded'] = [self.get_year_first_traded(t) for t in range(len(self.yh_tickers))]
        self.data['year_founded'] = [self.get_year_founded(t) for t in range(len(self.yh_tickers))]

    def ticker_info(self, ticker):
        """Return a dictionary of information about the given ticker. If the ticker is not found, return an empty dictionary."""
        ticker = ticker.upper()

        if ticker not in self.stocks_symbols:
            print("Ticker not found.")
            return {}

        t = Ticker(ticker)
        asset_p = t.asset_profile[ticker]
        quote_t = t.quote_type[ticker]
        if type(asset_p) != dict:
            asset_p = {}
        if type(quote_t) != dict:
            quote_t = {}

        return {**asset_p, **quote_t}
        
    def verify_dataset_downloaded(self):
        """
        Verify that the dataset has been downloaded and extracted correctly.
        
        If not, print an error message and set self.data to an empty dataframe."""
        error = ""
        directions = "\n Please ensure that you have downloaded the stock data first from https://www.kaggle.com/datasets/footballjoe789/us-stock-dataset. \n Then extract Stock/ to the data/ folder and move Stock_List to data/. \n Then run this script again. \n Thank you. \n"

        if os.path.exists("data/Stock_List.parquet") == False:
            error = "data/Stock_List.parquet is missing. "

        if os.path.exists("data/Stock/"):
            files_found = len(glob("data/Stock/*"))
            if files_found == 0:
                error = "No stock data found in data/Stock/ folder. "
        else:
            error = "data/Stock/ folder is missing. "
        
        if error != "":
            print(error + directions)
            self.data = pd.DataFrame()
            self.stocks_symbols = []
            self.yh_tickers = []
        return True if error == "" else False

def get_emotion_df(MODEL:str=f"cardiffnlp/twitter-xlm-roberta-base-sentiment", augment:bool=True, aug_n = 400000):
    """Parses the emotion dataframes and returns a dataframe.
    
    This saves a little bit of the processing time at the expense of storage space.
    
    Returns:
        pd.DataFrame"""
    if os.path.exists('data/Emotions/classes_df.parquet') and os.path.exists('data/Emotions/x_train_df.parquet') and os.path.exists('data/Emotions/y_train_df.parquet') and augment == False:
        #df_train = load_file('data/Emotions/df_train.parquet')
        classes = load_file('data/Emotions/classes_df.parquet').values.flatten().tolist()
        sentences_train = load_file('data/Emotions/x_train_df.parquet').values.flatten().tolist()
        sentences_test = load_file('data/Emotions/x_test_df.parquet').values.flatten().tolist()
        emotions_train = load_file('data/Emotions/y_train_df.parquet').values.flatten().tolist()
        emotions_test = load_file('data/Emotions/y_test_df.parquet').values.flatten().tolist()
        emotions_augs = []
        sentences_augs = []
    elif os.path.exists('data/Emotions/classes_df.parquet') and os.path.exists('data/Emotions/x_train_df.parquet') and os.path.exists('data/Emotions/y_train_df.parquet') and augment == True and os.path.exists('data/Emotions/x_augs_df.parquet') and os.path.exists('data/Emotions/y_augs_df.parquet'):
        classes = load_file('data/Emotions/classes_df.parquet').values.flatten().tolist()
        sentences_train = load_file('data/Emotions/x_train_df.parquet').values.flatten().tolist()
        sentences_test = load_file('data/Emotions/x_test_df.parquet').values.flatten().tolist()
        emotions_train = load_file('data/Emotions/y_train_df.parquet').values.flatten().tolist()
        emotions_test = load_file('data/Emotions/y_test_df.parquet').values.flatten().tolist()
        emotions_augs = load_file('data/Emotions/y_augs_df.parquet').values.flatten().tolist()
        sentences_augs = load_file('data/Emotions/x_augs_df.parquet').values.flatten().tolist()
    else:
        df_train, df_test, df_augs = parse_emotion_dataframes([0, 1, 2, 3, 4, 5], ensure_only_one_label=True, augment_data=augment, aug_n = aug_n)

        df_train = df_train.drop_duplicates(subset=['text'])
        df_test = df_test.drop_duplicates(subset=['text'])
        df_augs.drop_duplicates(subset=['text'])

        if not augment: sentences_augs, emotions_augs = [], []
        classes = df_train.columns[1:].values.tolist()
        sentences_test = df_test.text.values
        sentences_test = get_sentence_vectors(sentences_test, MODEL)
        sentences_train = df_train.text.values
        sentences_train = get_sentence_vectors(sentences_train, MODEL)

        if augment: 
            sentences_augs = df_augs.text.values
            sentences_augs = get_sentence_vectors(sentences_augs, MODEL)

        emotions_train = df_train.drop('text', axis=1, inplace=False).values.tolist()
        emotions_test = df_test.drop('text', axis=1, inplace=False).values.tolist()
        if augment: emotions_augs = df_augs.drop('text', axis=1, inplace=False).values.tolist()

        
        if not os.path.exists('data/Emotions/classes_df.parquet'): save_file(classes, 'data/Emotions/classes_df.parquet')
        if not os.path.exists('data/Emotions/x_df.parquet'): save_file(sentences_train, 'data/Emotions/x_train_df.parquet')
        if not os.path.exists('data/Emotions/y_df.parquet'): save_file(emotions_train, 'data/Emotions/y_train_df.parquet')
        if not os.path.exists('data/Emotions/x_test_df.parquet'): save_file(sentences_test, 'data/Emotions/x_test_df.parquet')
        if not os.path.exists('data/Emotions/y_test_df.parquet'): save_file(emotions_test, 'data/Emotions/y_test_df.parquet')
        if augment:
            if not os.path.exists('data/Emotions/y_augs_df.parquet'): save_file(emotions_augs, 'data/Emotions/y_augs_df.parquet')
            if not os.path.exists('data/Emotions/x_augs_df.parquet'): save_file(sentences_augs, 'data/Emotions/x_augs_df.parquet')
        

    return sentences_train, emotions_train, sentences_test, emotions_test, sentences_augs, emotions_augs, classes

class create_triplets():
    """Converts (x,y) to (anchor,positive,negative) where anchor and positive are positive examples and negative is a negative example. 
    
    This will also randomly select two random classes to select the positive and negative examples from.
    
    Function could probably be optimized to be faster. I welcome any suggestions.
    
    Args:
        x: np.array of shape (num_examples, x_shape)
        y: np.array of shape (num_examples, num_classes)
        batch_size: int, the number of examples to return in each batch
        shuffle: bool, whether to shuffle the data or not
        seed: int, the seed to use for the random number generator
        
    Returns:
        anchor: np.array of shape (batch_size, x_shape)
        positive: np.array of shape (batch_size, x_shape)
        negative: np.array of shape (batch_size, x_shape)
        anchor_class: np.array of shape (batch_size, num_classes)
        
    Example:
        triplets = create_triplets(x, y, batch_size=32, shuffle=True, seed=42)
        
        model_siamese.fit(triplets.generator(), steps_per_epoch=triplets.num_batches, epochs=10)
        """
    def __init__(self, x:torch.tensor, y:torch.tensor, batch_size:int=32, shuffle:bool=True, seed:int=42):
        self.i = 0
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.batch_range = range(self.batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.indices = torch.arange(self.x.shape[0])
        self.num_classes = self.y.shape[1]
        self.class_indices = [torch.where(self.y[:,i] == 1)[0] for i in range(self.num_classes)] # this is the indices for all the examples in each class
        self.num_examples_per_class = [len(x) for x in self.class_indices]
        self.num_examples = sum(self.num_examples_per_class)
        self.num_batches = self.num_examples // self.batch_size
        self.num_examples = self.num_batches * self.batch_size
        self.num_examples_per_class = [x // self.batch_size for x in self.num_examples_per_class]
        if seed != None: torch.manual_seed(seed)

        if self.shuffle:
            for i in range(self.num_classes):
                self.class_indices[i] = torch.randperm(self.class_indices[i].shape[0])
            self.class_indices = [self.indices[x] for x in self.class_indices]

    def __iter__(self):
        return self

    def get_batch(self): 
        # randomly select two classes
        anchor_class = torch.randint(self.num_classes, (self.batch_size,))

        # make sure the contrasting class is not the same as the anchor class
        contrasting_class = torch.randint(self.num_classes - 1, (self.batch_size,))
        contrasting_class[contrasting_class >= anchor_class] += 1 

        # randomly select batch examples from each class
        examples_anchor   = [torch.randint(self.num_examples_per_class[anchor_class[i]], (1,)) for i in self.batch_range]
        examples_positive = [torch.randint(self.num_examples_per_class[anchor_class[i]], (1,)) for i in self.batch_range]
        examples_negative = [torch.randint(self.num_examples_per_class[contrasting_class[i]], (1,)) for i in self.batch_range]
        
        # get the indices for the examples
        indexes_anchor =   [self.class_indices[anchor_class[i]][examples_anchor[i]] for i in self.batch_range]
        indexes_positive = [self.class_indices[anchor_class[i]][examples_positive[i]] for i in self.batch_range]
        indexes_negative = [self.class_indices[contrasting_class[i]][examples_negative[i]] for i in self.batch_range]
        
        anchor = self.x[indexes_anchor]
        positive = self.x[indexes_positive]
        negative = self.x[indexes_negative]

        return [anchor, positive, negative], anchor_class

    def generator(self):
        for _ in range(self.num_batches):
            yield self.get_batch()

def available_mem():
    """Return the available GPU memory in GB."""
    MB_memory = int("".join([x for x in subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv"]).decode() if x.isdigit()]))
    GB_memory = MB_memory / 1000
    return GB_memory

def get_datasets(kaggle_api_key, data_nasdaq_key=None):
    """Download the datasets from kaggle and then convert thhem to highly compressed parquet files.
    
    Args:
        kaggle_api_key: str, the kaggle api key
        data_nasdaq_key: str, the data.nasdaq.com api key... not used currently"""
    username, password = kaggle_api_key.split(' ')
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = password
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

    # download the various kaggle datasets
    download_dataset(
            'https://www.kaggle.com/datasets/footballjoe789/us-stock-dataset', 
            kaggle_api_key, 
            files_to_move={'us-stock-dataset/Data/Stocks': 'Stocks'}, #, 'us-stock-dataset/Stock_List.csv': 'Stock_List.csv'
            delete=True,
            dest_name='Stock',
            min_files_expected=100
            )

    download_dataset(
            'https://www.kaggle.com/datasets/sarthmirashi07/us-macroeconomic-data', 
            kaggle_api_key, 
            files_to_move={'US_macroeconomics.csv': 'macro/US_macroeconomics.csv'},
            delete=True,
            dest_name='Macro',
            min_files_expected=1
            )

    download_dataset(
            'https://www.kaggle.com/datasets/mathurinache/goemotions',
            kaggle_api_key,
            files_to_move={'goemotions.csv': 'Emotions/goemotions.csv'},
            delete=True,
            dest_name='Emotions',
            min_files_expected=1
            )

    download_dataset(
            'https://www.kaggle.com/datasets/parulpandey/emotion-dataset',
            kaggle_api_key,
            files_to_move={'training.csv': 'Emotions/training.csv', 'validation.csv': 'Emotions/validation.csv', 'test.csv': 'Emotions/test.csv'},
            delete=True,
            dest_name='Emotions',
            min_files_expected=3
            )

    download_dataset(
            'https://www.kaggle.com/datasets/kosweet/cleaned-emotion-extraction-dataset-from-twitter',
            kaggle_api_key,
            files_to_move={'dataset(clean).csv': 'Emotions/dataset(clean).csv'},
            delete=True,
            dest_name='Emotions',
            min_files_expected=1
            )

    download_dataset(
            'https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests',
            kaggle_api_key,
            files_to_move={'raw_partner_headlines.csv': 'Text/raw_partner_headlines.csv', 'raw_analyst_ratings.csv': 'Text/raw_analyst_ratings.csv', 'analyst_ratings_processed.csv': 'Text/analyst_ratings_processed.csv'},
            delete=True,
            dest_name='Text',
            min_files_expected=3)

    if not os.path.exists('data/Emotions/dailydialog.parquet'):
        print('Downloading DailyDialog dataset...')
        download_file('http://yanran.li/files/ijcnlp_dailydialog.zip', 'ijcnlp_dailydialog.zip', 'data/Emotions/')

        # extract the zip file
        with zipfile.ZipFile('data/Emotions/ijcnlp_dailydialog.zip', 'r') as zip_ref:
            zip_ref.extractall('data/Emotions/')
        for f in os.listdir('data/Emotions/ijcnlp_dailydialog/'):
            if '.zip' in f or 'act' in f or 'topic' in f or 'readme' in f:
                os.remove('data/Emotions/ijcnlp_dailydialog/'+f)
        # read data\Emotions\ijcnlp_dailydialog\dialogues_text.txt, remove \n, split('__eou__')
        with open('data\Emotions\ijcnlp_dailydialog\dialogues_emotion.txt', 'r', encoding='utf-8') as f:
            labels = [{ 0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}[int(x)] for x in f.read().replace('\n', ' ').split(' ') if x.isnumeric()]
            # check that x is numeric
        with open('data/Emotions/ijcnlp_dailydialog/dialogues_text.txt', 'r', encoding='utf-8') as f:
            text = f.read().replace('\n', '').replace('.', '').replace(' ’ ', r"'").replace(' , ', r", ").replace(' ? ', '? ').replace(' ! ', '! ').replace(' : ', ': ').replace(' ; ', '; ').replace('  ', ' ')
            # remove trailing spaces
        text = [x.strip() for x in text.split('__eou__') if x.strip() != '']
        assert len(text) == len(labels)
        df = pd.DataFrame({'text': text, 'label': labels})
        save_file(df, 'data/Emotions/dailydialog.parquet')
        shutil.rmtree('data/Emotions/ijcnlp_dailydialog')
        os.remove('data/Emotions/ijcnlp_dailydialog.zip')
        del df, text, labels
    
    print('Converting project files to parquet')
    convert_project_files_to_parquet()

    # clear the username and key from the environment variables
    os.environ['KAGGLE_USERNAME'] = "" 
    os.environ['KAGGLE_KEY'] = ""

# Preprocess text (username and link placeholders)
def preprocess(text):
    """The preprocessing function used to preprocess the text for the finAlbert model.
    
    Args:
        text (str): The text to preprocess.
        
    Returns:
        str: The preprocessed text."""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentence_vectors(sentences, MODEL):
    """Returns the sentence vectors for the given sentences. The sentence vectors are calculated using the finAlbert tokenizer.
    
    Args:
        sentences (list-like): A list of sentences.
        MODEL (str): The model to use (finAlbert).
        """
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    return [tokenizer.encode(preprocess(sentence), add_special_tokens=True, max_length=128, truncation=True, padding="max_length") for sentence in sentences]

def stock_missing_days(ticker, start_date=None, remove_days_in_between_datapoints=True, exchange='NYSE', year_format='YYYY-MM-DD'):
    """ Returns a list of missing days for a given stock ticker. If start_date is None, it will use the earliest date in the dataset.

    Args:
        ticker: str
        start_date: str in YYYY-MM-DD format
        remove_days_in_between_datapoints: bool
        exchange: str
        year_format: str

    Returns:
        missing_yyyymmdd: list of str
    """
    #nyse = mcal.get_calendar(exchange)
    stock = load_file(f'data\Stock\{ticker}.parquet')
    if start_date == None: start_date = arrow.get(stock.date.min()).format(year_format)
    todays_date = arrow.now().format(year_format)
    if remove_days_in_between_datapoints: start_date = arrow.get(stock.date.max()).format(year_format)

    all_yyyymmdd = [x.format(year_format) for x in arrow.Arrow.range('day', arrow.get(start_date), arrow.get(todays_date))]
    #off = [arrow.get(x).format(year_format) for x in mcal.date_range(nyse.schedule(start_date=start_date, end_date=todays_date), frequency='1D')]
    all_yyyymmdd = [x for x in all_yyyymmdd if arrow.get(x).weekday() < 5]
    not_missing_yyyymmdd = [arrow.get(x).format(year_format) for x in stock.date.tolist()]
    missing_yyyymmdd = [x for x in all_yyyymmdd if x not in not_missing_yyyymmdd]
    missing_yyyymmdd = [arrow.get(x).format('YYYY-MM-DD') for x in missing_yyyymmdd]
    
    return missing_yyyymmdd

def update_stock_data(ticker=None, missing_yyyymmdd=None, exchange='NYSE', save=True):
    """ Updates the stock data for a given ticker with the missing days in the dataset.

    Args:
        ticker: str, list of str, or None (if None, will update all stocks in data\Stock)
        missing_yyyymmdd: list of str
        exchange: str (default='NYSE', not used yet)
        save: bool

    Returns:
        stock: pd.DataFrame
    """
    if ticker == None: 
        tickers = [x.split('.')[0] for x in os.listdir('data\Stock') if x.endswith('.parquet')]
    elif type(ticker) == str:
        tickers = [ticker]

    with tqdm(total=len(tickers)) as pbar:
        for ticker in tickers:
            if missing_yyyymmdd == None: 
                missing_yyyymmdd = stock_missing_days(ticker, exchange=exchange)
            stock = load_file(f'data\Stock\{ticker}.parquet').drop(columns=['adjclose', 'symbol', 'dividends', 'stock splits'], inplace=False, errors='ignore')
                
            if len(missing_yyyymmdd) <= 1: 
                pbar.desc = f'{ticker} is up to date.'
                pbar.update(1)
                continue
            else:
                pbar.desc = f'Updating {ticker}... missing {len(missing_yyyymmdd)} days. {missing_yyyymmdd}'
                pbar.update(1)

            t = Ticker(ticker)
            stock_data = t.history(period='max', interval='1d', start=missing_yyyymmdd[0], end=missing_yyyymmdd[-1])
            stock_data = stock_data.reset_index(drop=False, inplace=False).drop(columns=['adjclose', 'symbol', 'dividends'], inplace=False, errors='ignore')
            
            
            stock = pd.concat([stock, stock_data], axis=0).drop_duplicates(inplace=False).reset_index(drop=True, inplace=False)
            #stock.date = stock.date.apply(lambda x: arrow.get(x).format('YYYY-MM-DD'))

            if save:
                try:
                    os.rename(f'data\Stock\{ticker}.parquet', f'data\Stock\{ticker}_old.parquet')
                    save_file(stock, f'data\Stock\{ticker}.parquet')
                    os.remove(f'data\Stock\{ticker}_old.parquet')
                except Exception as e:
                    os.rename(f'data\Stock\{ticker}_old.parquet', f'data\Stock\{ticker}.parquet')
                    print(f'Failed to update {ticker}')
                    import traceback
                    traceback.print_exc()

                    raise e
            else:
                return stock

def download_file(url, filename, move_to=None):
    """Downloads a file from a url and saves it to the specified filename. Not used in the current version but useful on windows, where this sometimes is needed as an alternative to wget.
    
    Args:
        url (str): The url to download the file from.
        filename (str): The filename to save the file to.
    """
    
    if not os.path.exists(move_to + filename) and not os.path.exists(move_to + filename.replace(filename.split('.')[-1], 'parquet')):
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            data = response.read() # a `bytes` object
            out_file.write(data)

    if move_to is not None:
        if not os.path.exists(os.path.dirname(move_to)):
            os.makedirs(os.path.dirname(move_to), exist_ok=True)
        if not os.path.exists(move_to + filename) and not os.path.exists(move_to + filename.replace(filename.split('.')[-1], 'parquet')):
            shutil.move(filename, move_to)

def daily_stats(df):
    """Takes in a dataframe and returns the daily std, mean, outliers, and count of the dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be grouped by date
        
        Returns
        -------
        pd.DataFrame
            The dataframe with the daily stats
    """
    df = df.drop(['text', 'stock'], axis=1, errors='ignore')
    df = pd.concat([df, pd.DataFrame(df.emotion.tolist(), index=df.index)], axis=1)
    df = df.drop(columns=['emotion'])

    # group by date
    df = df.groupby('date').agg([lambda x: x.std() if x.count() > 1 else 0, lambda x: x.mean() if x.count() > 1 else 0, lambda x: x.count() if x.count() > 1 else 0])
    # flatten the columns
    df.columns = ['_'.join([str(x).replace('<lambda_0>', 'std').replace('<lambda_1>', 'mean').replace('<lambda_2>', 'count').replace('<lambda_3>', 'occured') for x in col]) for col in df.columns.values]
    df.reset_index(inplace=True)
    return df
    

def scrape_tweets(since='2019-11-01', until='2020-03-30', max_tweets=20, update_twitter_data=True, co_list=['#kr', 'the kroger co.', 'william rodney mcmullen', 'grocery stores']):
    """
        Scrape tweets between since and until dates containing specified search terms. Returns max_tweets number of tweets for each search term.
        ***Currently only searches based on company "shortName" from "company_list.pkl". This can be updated in the future by passing a list of
        search terms into the funtion, adding more keys to search for (ex: ticker symbol, CEO/COO names, etc), or accessing a separate search terms file.***

    Args:
        since (str): Starting date (YYYY-MM-DD) for tweet queries. If update_twitter_data is True, will use the last date in the existing data.
        until (str): Ending date (YYYY-MM-DD) for tweet queries. If update_twitter_data is True, will use the current date if since>until.
        max_tweets (int): Maximum number of tweets to return per search term.
        update_twitter_data (bool): If True, will update the twitter data for each search term. If False, will only return the existing data.
        co_list (list): List of search terms to use for twitter queries. 
    """
    assert type(co_list) == list, 'co_list must be a list of strings.'
    assert type(since) == str, 'since must be a string in the format YYYY-MM-DD.'
    assert type(until) == str, 'until must be a string in the format YYYY-MM-DD.'
    assert type(max_tweets) == int, 'max_tweets must be an integer.'
    assert type(update_twitter_data) == bool, 'update_twitter_data must be a boolean.'

    try:
        if not os.path.exists('data/Twitter'):
            os.makedirs('data/Twitter', exist_ok=True)

        clean_list = co_list

        tweets_list = []
        counter = 0
        date_range = arrow.Arrow.range('day', arrow.get(since), arrow.get(until))
        range_tuples = [(arrow.get(x).format('YYYY-MM-DD'), arrow.get(x).shift(days=1).format('YYYY-MM-DD')) for x in date_range]

        # iterate through cleaned company_list
        with tqdm(total=len(clean_list) * len(range_tuples)) as pbar:
            unique_search_terms = "___" + "---".join(co_list)
            if os.path.exists(f'./data/Twitter/twitter_data{unique_search_terms}.parquet'):
                df = load_file(f'./data/Twitter/twitter_data{unique_search_terms}.parquet')
                last_date = arrow.get(df['date'].apply(lambda x: arrow.get(x).date()).max())
                first_date = arrow.get(df['date'].apply(lambda x: arrow.get(x).date()).min())
                days_already_scraped = [arrow.get(x).format('YYYY-MM-DD') for x in df.date.unique().tolist()]
                range_tuples = [(x[0], x[1]) for x in range_tuples if x[0] not in days_already_scraped]
                pbar.total = len(range_tuples) * len(co_list)
                if len(range_tuples) == 0:
                    print(f"Twitter data is up to date. Last date: {last_date.format('YYYY-MM-DD')}")
                    return
                if last_date >= arrow.get(since):
                    # today's date
                    today = arrow.now().date()
                    if last_date == today:
                        print(f"Twitter data is up to date. Last date: {last_date.format('YYYY-MM-DD')}")
                        return
                    else:
                        since = last_date.shift(days=1).format('YYYY-MM-DD')
                        print(f"Updating Twitter data from {since} to {until}.")

            for name in co_list:
                # scrape twitter data based on search terms and dates, return up to max_tweets for each search
                # get daily intervals for date range using arrow

                for since_, until_ in range_tuples:
                    for i,tweet in enumerate(sntw.TwitterSearchScraper(name + ' since:' + since_ + ' until:' + until_).get_items()):
                        if i > max_tweets:
                            break
                        if len(tweet.rawContent) > 0:
                            tweets_list.append([tweet.date, tweet.rawContent, tweet.user.username, name])
                            # twitter limits query rate, the counter simply gives an idea on progress
                            pbar.desc = f"Scraping progress: found {len(tweets_list)} tweets for {name} on {since_}"
                            pbar.update(1)
                            counter += 1
                        else:
                            pbar.desc = f"Scraping progress: no tweets found for {name} on {since_}"
                            pbar.update(1)
                            counter += 1

                        sleep(random.random() * random.randint(2, 4))
                
            # convert scraped tweets to dataframe
            tweets_df = pd.DataFrame(tweets_list, columns=['date', 'text', 'username', 'searchterm'])
            tweets_df = tweets_df.drop_duplicates(inplace=False, subset=['date', 'text', 'username', 'searchterm']).reset_index(drop=True, inplace=False).dropna(inplace=False)

            # if updating existing twitter data append to existing file (if it exists), else save as new file
            if os.path.exists(f'./data/Twitter/twitter_data{unique_search_terms}.parquet'):
                if update_twitter_data:
                    #old_twitter_data = pd.read_parquet('./data/Twitter/twitter_data.parquet')
                    old_twitter_data = load_file(f'./data/Twitter/twitter_data{unique_search_terms}.parquet')
                    new_twitter_data = pd.concat([old_twitter_data, tweets_df], ignore_index=True)
                    new_twitter_data = new_twitter_data.drop_duplicates(inplace=False, subset=['date', 'text', 'username', 'searchterm'])
                    #new_twitter_data.to_parquet(path='./data/Twitter/twitter_data.parquet', engine='pyarrow', compression='brotli')
                    save_file(new_twitter_data, f'./data/Twitter/twitter_data{unique_search_terms}.parquet')
                else:
                    #tweets_df.to_parquet(path='./data/Twitter/twitter_data.parquet', engine='pyarrow', compression='brotli')  
                    save_file(tweets_df, f'./data/Twitter/twitter_data{unique_search_terms}.parquet')       
            else:
                #tweets_df.to_parquet(path='./data/Twitter/twitter_data.parquet', engine='pyarrow', compression='brotli')
                save_file(tweets_df, f'./data/Twitter/twitter_data{unique_search_terms}.parquet')
    except Exception as e:
        print(e)
        return tweets_list
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Attempting to save recovery file else return partial data.")
        return tweets_list

def merge_data_timeline(company_emotion_stats, stock, generic_news_stats, fillzero=True, impute=True):
        """
        Merges the data from the company emotion stats and the generic news stats into a single dataframe

        Also performs feature engineering on the data by normalizing the data and creating columns for time features
        
        Parameters
        ----------
        company_emotion_stats : pandas.DataFrame
            The dataframe containing the company emotion stats
        generic_news_stats : pandas.DataFrame
            The dataframe containing the generic news stats
        fillzero : bool, optional
            Whether to fill the NaN values with 0, by default True
        impute : bool, optional
            Whether to impute the NaN values (missing days), by default True
        
        Returns
        -------
        pandas.DataFrame
            The merged dataframe by date, with zeros for all values where there is no intersection 
        """
        company_emotion_stats.date = company_emotion_stats.date.apply(lambda x: arrow.get(str(x)[:10]).format('YYYY-MM-DD') if len(str(x)) > 7 else None)
        generic_news_stats.date = generic_news_stats.date.apply(lambda x: arrow.get(str(x)[:10]).format('YYYY-MM-DD') if len(str(x)) > 7 else None)
        df = pd.merge(company_emotion_stats, generic_news_stats, how='outer', on='date')
        df = df.sort_values(by='date', inplace=False)
        df = df.dropna(subset=['date'])

        df, stock = intersect_df(df, stock)
        df = pd.merge(df, stock, how='outer', on='date')
        df = df.sort_values(by='date', inplace=False).reset_index(drop=True, inplace=False)
        df['day_of_week'] = df['date'].apply(lambda x: arrow.get(x).weekday())
        df['day_of_month'] = df['date'].apply(lambda x: arrow.get(x).day)
        df['month'] = df['date'].apply(lambda x: arrow.get(x).month)
        df['mon_or_fri'] = df['day_of_week'].apply(lambda x: 1 if x == 0 or x == 4 else 0)
        # diff between high and low
        df['range'] = df['high'] - df['low']
        # percent change
        df['open'] = df['open'].pct_change()
        df['high'] = df['high'].pct_change()
        df['low'] = df['low'].pct_change()
        df['volume'] = df['volume'].pct_change()
        df['close'] = df['close'].pct_change()
        df['range'] = df['range'].pct_change()
        if fillzero:
            df = df.fillna(0)
        if impute:
            df = df.interpolate(method='linear')
        
        return  df

class sliding_window_view_generator():
    """
    Creates a sliding window view of the input array. Pads the input array with zeros to make sure the sliding window view is the same size as the input array.
    
    Parameters
    ----------
    x : torch.Tensor
        The input array
    window_size : int
        The size of the sliding window
    
    Yields
    -------
    numpy.ndarray
        The sliding window view of the input array
    """
    def __init__(self, x, window_size=32, device='cuda'):
        self.x = x
        self.x = torch.from_numpy(self.x.astype(np.float32))
        # replace nan with 0
        self.x = torch.where(torch.isnan(self.x), torch.zeros_like(self.x), self.x)
        # replace inf with 1
        self.x = torch.where(torch.isinf(self.x), torch.ones_like(self.x), self.x)
        self.x = self.x.to(device)
        self.x = torch.nn.functional.pad(self.x, (window_size, 0), 'constant', 0)
        self.window_size = window_size
        self.current_index = 0
        if self.x[0].shape[0] > 2:
            self.shape = (len(self.x),) + (self.x[0].shape[0],) + self.x[0].shape[1:]
        else:
            self.shape = (len(self.x),) + (self.x[0].shape[0],) + (self.window_size,) + self.x[0].shape[1:]
        
    def __iter__(self):
        return self

    def next(self):
        self.current_index += 1
        if self.current_index >= len(self.x):
            self.current_index = 0
        if self.x[0].shape[0] > 2:
            yield self.x[self.current_index:self.current_index+self.window_size]
        else:
            yield self.x[self.current_index:self.current_index+self.window_size].unsqueeze(0).permute(0,2,1,3)