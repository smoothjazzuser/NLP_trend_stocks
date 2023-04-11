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
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
cuda = torch.device("cuda")
cpu = torch.device("cpu")

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
    """"convert the files to parquet format, which is a much better for this project. Will convert all csv, xlsx, txt, json, dat and pkl files in data/ to parquet and then delete the original files (to save space)."""
    for file_type in ['csv', 'xlsx', 'txt', 'json', 'dat', 'pkl']:
        files_to_compress = glob(f'data/*/*.{file_type}') + glob(f'data/*/*/*.{file_type}') + glob(f'data/*.{file_type}')
        for file in files_to_compress:
            if not os.path.exists(file.replace(f'.{file_type}', '.parquet')):
                df = load_file(file)
                if type(df) != bool: df.to_parquet(file.replace(f'.{file_type}', '.parquet'), compression='brotli', engine='pyarrow')
            if os.path.exists(file):
                os.remove(file)
    # remove npy files
    for file in glob('data/*.npy'):
        os.remove(file)

def load_file(file:str):
    """Reads file into a pandas dataframe. Supports csv, xlsx, txt, json, lz4, dat and pkl files.
    
    args:
        file path (str): The file to load."""
    file_type = file.split('.')[-1]
    if file_type == 'parquet':
        return pd.read_parquet(file)
    elif file_type == 'csv':
        return pd.read_csv(file, low_memory=False, parse_dates=True, infer_datetime_format=True, on_bad_lines='skip', encoding_errors= 'replace')
    elif file_type == 'xlsx':
        return pd.read_excel(file, parse_dates=True)
    elif file_type == 'pkl':
        return load(file)
    elif file_type in ['.lz4', 'lz4']:
        return load(file, compression='lz4')
    elif file_type == 'json':
        return pd.read_json(file)
    elif file_type == 'txt':
        return pd.read_csv(file, sep='\t', low_memory=False, parse_dates=True, infer_datetime_format=True, on_bad_lines='skip', encoding_errors= 'replace')
    elif file_type == 'dat':
        return pd.read_csv(file, sep='\t', low_memory=False, parse_dates=True, infer_datetime_format=True, on_bad_lines='skip', encoding_errors= 'replace')
    else:
        try:
            return load(file)
        except:
            os.remove(file)
            return False

def save_file(df, file:str, force_type=None):
    """Saves a pandas dataframe to a file. Supports parquet, pkl, csv and xlsx files.
    
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
        df.to_parquet(file, compression='brotli', engine='pyarrow')
    elif file_type == 'pkl':
        dump(df, file + '.lz4', compression='lz4')
    elif file_type == 'csv':
        df.to_csv(file, index=False)
    elif file_type == 'xlsx':
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
    """Augments a list of sentences until it reaches a certain length."""
    
    tagger = Classifier.load('sentiment')
    
    def sentiment(l):
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
    
    sentences = sentiment(l)
    aug_pre = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", aug_min=1, aug_max=2, device='cuda')
    aug_pre2 = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
    

    aug = naf.Sometimes([
        naw.SynonymAug(aug_src='wordnet', aug_p=0.05, aug_min=0, aug_max=2),
        nac.RandomCharAug(action="insert", aug_char_min=0, aug_char_max=2, aug_word_min=0, aug_word_max=2, aug_word_p=0.05, aug_char_p=0.05),
        nac.KeyboardAug(aug_char_p=0.05, aug_word_p=0.3, aug_word_min=0, aug_word_max=2, aug_char_min=0, aug_char_max=2),
        naw.SpellingAug(aug_min=0, aug_max=3, aug_p=0.05),
        ])

    print(f'step 1: augmenting len(list) {len(l)} to {maxlength//4} using naw.ContextualWordEmbsAug')
    with tqdm(total=maxlength, dynamic_ncols=True) as pbar:
        l_frozen_size = len(l)
        augments = set(l)
        if len(l) < maxlength//4:
            n = max(1, maxlength//4//len(l))
            
            for i in range(l_frozen_size):
                potential1 = aug_pre.augment(l[i], n=n, num_thread=min(n, num_thread)) 
                pot_senti1 = sentiment(potential1)
                pot_senti1 = [potential1[j] for j in range(len(potential1)) if pot_senti1[j] == sentences[i]]

                potential2 = aug_pre2.augment(l[i], n=n, num_thread=1)
                pot_senti2 = sentiment(potential2)
                pot_senti2 = [potential2[j] for j in range(len(potential2)) if pot_senti2[j] == sentences[i]]

                augments.update(pot_senti1)
                augments.update(pot_senti2)

                pbar.update(len(augments) - pbar.n)
                if len(augments) >= maxlength//4:
                    break
            l = [x for x in augments]
        
        with contextlib.suppress(Exception): del sentences, aug_pre, aug_pre2, pot_senti1, pot_senti2, potential1, potential2, augments, tagger
            
            
        print(f'step 2: augmenting len(list) {len(l)} to {maxlength} using synonyms and spelling errors')
        augments = set()
        pbar_start = pbar.n
        length = maxlength - len(l)
        n = max(length // len(l) - 1, 1)
        if n > 0 and length > 0:
            
                while len(augments) < length:
                    for i in range(len(l)):
                        augs = aug.augment(l[i], n=n, num_thread=num_thread)
                        while l[i] in augs:
                            augs.remove(l[i])
                        augments.update(augs)
                        pbar.update(len(augments) - pbar.n + pbar_start)
                        if len(augments) >= length:
                            break
        else:
            return [x for x in augments]
    # shuffle(augments)
    augments = [x for x in augments]
    augments = random.sample(augments, len(augments))

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.suppress(Exception):
            del aug, aug_pre, aug_pre2, augs 
        collect()
    return augments[:length]

def parse_emotion_dataframes(selection: int = [0, 1, 2, 3, 4], ensure_only_one_label: bool = True, augment_data=True, aug_n = 400000):
    """Parses the emotion dataframes. Normalizes each of the specific 29 dataset labels to the same label names, then combines the five of them into one dataframe. The resulting dataset is highly unbalanced, with predominant negative emotions.
    
    Args: 
        selection (int): The emotion dataframes to parse. 0 = https://www.kaggle.com/datasets/mathurinache/goemotions, [1, 2, 3] = www.kaggle.com/datasets/parulpandey/emotion-dataset, 4 = https://www.kaggle.com/datasets/kosweet/cleaned-emotion-extraction-dataset-from-twitter
        ensure_only_one_label (bool): If True, ensures that each row only has one label. If False, allows multiple labels per row.

    Returns:
        df (pd.DataFrame): The combined dataframe of the selected datasets.
    """

    def df_load_(file, drop_cols, new_cols, rename_cols=[]):
        """Loads the dataframe and drops the columns that are not needed."""
        df0, df1, df2, df3, df4 = pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
        df = load_file(file)
        if 'label' in df.columns:
            df = pd.get_dummies(df, columns=['label'])
        if 'Emotion' in df.columns:
            df = pd.get_dummies(df, columns=['Emotion'])
        df = df.drop(columns=drop_cols)
        if rename_cols.__len__() > 0:
            df.columns = rename_cols
        for col in new_cols:
            if col not in df.columns:
                df[col] = 0
        all_emotions = list(set([x for x in df.columns[df.columns != 'text'] if x not in drop_cols]))


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


    df = pd.concat([df0, df1, df2, df3, df4], ignore_index=True)
    df = df.drop_duplicates(subset=['text'])
    df = df.reset_index(drop=True)
    df.fillna(0, inplace=True)
    df_augs = pd.DataFrame(columns=df.columns)

    if augment_data:
        subset_lists_per_label = {df.columns[1:].tolist()[i]: df[df.columns[1:].tolist()[i]] == 1 for i in range(len(df.columns[1:].tolist()))}
        for label in subset_lists_per_label:
            subset_to_add = df[subset_lists_per_label[label]].text.values.tolist()

            subset_to_add = augment_list_till_length(subset_to_add, num_thread=12, maxlength=aug_n)
            subset_labels = [1]* len(subset_to_add)

            subset = pd.DataFrame({'text': subset_to_add, label: subset_labels})
            df_augs = pd.concat([df_augs, subset], ignore_index=True)
            df_augs.fillna(0, inplace=True)
            df_augs.drop_duplicates(subset=['text'], inplace=True)

            df_augs = df_augs[~df_augs.text.isin(df.text.values.tolist())]
            df_augs = df_augs.reset_index(drop=True)

    return df, df_augs

def df_to_datetime(df: pd.DataFrame, offset_days: int = 0):
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
        df[col] = df[col].apply(lambda x: x.split(' ')[0])
        offset_days = pd.Timedelta(f'{offset_days} days') #pd.Timedelta(days=offset_days)
        time = pd.to_datetime(df[col], format='%Y-%m-%d')
        time = time.apply(lambda x: x + offset_days)
        # convert to days
        df.index = time

        print(time)
        df.drop(columns=[col], inplace=True)
    return df

def interpolate_months_to_days(df: pd.DataFrame, extend_trend_to_today: bool = False):
    """Interpolates the dataframe to account for missing days. E.g, Macro data is usually available on a monthly basis.
    
    Args: DataFrame, extend_trend_to_today (bool): If True, the index will be extended to today (only enabled for monthly data).)"""
    #check if index is already datetime
    df = df_to_datetime(df)

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

def intersect_df(df1: pd.DataFrame, df2: pd.DataFrame, interpolate_to_days: bool = False, extend_trend_to_today: bool = False, offset_2nd_df_by_days: int = 1):
    """Performantly Intersects two dataframes based on their (datetime) index.
    
    Args: 
        - Two dataframes with a datetime index.
        - interpolate_to_days (bool): If True, the dataframes will be interpolated to account for missing days. E.g, Macro data is usually available on a monthly basis.
        - extend_trend_to_today (bool): If True, the index will be extended to today (only enabled for monthly data).
        - offset_2nd_df_by_days (int): The number of days to offset the second dataframe by n. This is useful for next day prediction.
    
    Returns: The datapoints shared between the two datasets."""

    if offset_2nd_df_by_days != 0:
        df2 = df_to_datetime(df2, offset_days=offset_2nd_df_by_days)


    if interpolate_to_days:
        df1 = interpolate_months_to_days(df1, extend_trend_to_today)
        df2 = interpolate_months_to_days(df2, extend_trend_to_today)

    df1 = df1[df1.index.isin(df2.index)]
    df2 = df2[df2.index.isin(df1.index)]

    return df1, df2
class get_macroeconomic_data ():
    """Aquire historical macroeconomic data from different sources."""
    def __init__(self, path):
        self.path = path

    def read_parquet(self):
        pass

    def get_FRED(self):
        pass

    def get_BLS(self):
        pass

    def get_BIS(self):
        pass

    def get_GDP(self):
        pass

    def get_inflation(self):
        pass

    def get_interest_rates(self):
        pass

    def get_unemployment(self):
        pass

    def get_exchange_rates(self):
        pass

    def get_commodities(self):
        pass

    def parse_for_trade_wars(self):
        pass

    def get_trade_wars(self):
        pass

    def get_nat_disasters(self):
        pass

    def calculate_average_weather(self):
        pass

    def get_average_weather(self):
        pass

    def get_holidays(self):
        pass

    def get_federal_holidays(self):
        pass

    def calculate_indicators(self):
        pass

    def get_indicators(self):
        pass

    def get_pandemics(self):
        pass
class aquire_stock_search_terms():
    """
    Gather the company info for all the ticker symbols and return a dataframe with relevant search terms for each company.

    If the stocks dataset is updated on kaggle, compank_list.pkl needs to be deleted and this run again if the symbols have changed. 

    TODO: It would be more efficient to manually pull the new stock data ourselves and keep the old ticker symbols.
    """
    def __init__(self, file_path = 'data/Stock/', file_ext = '.parquet'):
        self.file_path = file_path
        self.file_ext = file_ext

        if self.verify_dataset_downloaded():
            self.load_symbols()
            self.ticker_list_to_dataframe()
            self.get_company_list()
            self.process_information()

    def load_symbols(self):
        """Load all the stock symbols."""
        self.stocks_symbols = load_file("data/Stock_List.parquet")['Symbol'].tolist()
        return self.stocks_symbols

    def ticker_list_to_dataframe(self):
        self.data = pd.DataFrame(self.stocks_symbols, columns = ['Ticker'])

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
        self.data['Company'] = [self.get_company(t) for t in range(len(self.yh_tickers))]
        self.data["Top_Executive"] = [self.get_company_officers(t) for t in range(len(self.yh_tickers))] 
        self.data['Industry'] = [self.get_industry(t) for t in range(len(self.yh_tickers))]
        self.data['quoteType'] = [self.get_quote_type(t) for t in range(len(self.yh_tickers))]
        self.data['sector'] = [self.get_sector(t) for t in range(len(self.yh_tickers))]
        self.data['first_traded'] = [self.get_year_first_traded(t) for t in range(len(self.yh_tickers))]
        self.data['year_founded'] = [self.get_year_founded(t) for t in range(len(self.yh_tickers))]

    def ticker_info(self, ticker):
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
    if os.path.exists('data/Emotions/classes_df.parquet') and os.path.exists('data/Emotions/x_df.parquet') and os.path.exists('data/Emotions/y_df.parquet') and augment == False:
        #emotion_df = load_file('data/Emotions/emotion_df.parquet')
        classes = load_file('data/Emotions/classes_df.parquet').values.flatten().tolist()
        sentences = load_file('data/Emotions/x_df.parquet').values.flatten().tolist()
        emotions = load_file('data/Emotions/y_df.parquet').values.flatten().tolist()
        emotions_augs = []
        sentences_augs = []
    elif os.path.exists('data/Emotions/classes_df.parquet') and os.path.exists('data/Emotions/x_df.parquet') and os.path.exists('data/Emotions/y_df.parquet') and augment == True and os.path.exists('data/Emotions/x_augs_df.parquet') and os.path.exists('data/Emotions/y_augs_df.parquet'):
        classes = load_file('data/Emotions/classes_df.parquet').values.flatten().tolist()
        sentences = load_file('data/Emotions/x_df.parquet').values.flatten().tolist()
        emotions = load_file('data/Emotions/y_df.parquet').values.flatten().tolist()
        emotions_augs = load_file('data/Emotions/y_augs_df.parquet').values.flatten().tolist()
        sentences_augs = load_file('data/Emotions/x_augs_df.parquet').values.flatten().tolist()
    else:
        emotion_df, df_augs = parse_emotion_dataframes([0, 1, 2, 3, 4], ensure_only_one_label=True, augment_data=augment, aug_n = aug_n)

        emotion_df = emotion_df.drop_duplicates(subset=['text'])
        df_augs = df_augs.drop_duplicates(subset=['text'])

        emotion_df.fillna(0, inplace=True)
        df_augs.fillna(0,inplace=True)
        classes = emotion_df.columns[1:].values.tolist()

        sentences = emotion_df.text.values
        sentences = get_sentence_vectors(sentences, MODEL)
        sentences_augs = df_augs.text.values
        if augment:
            sentences_augs = get_sentence_vectors(sentences_augs, MODEL)

        emotions = emotion_df.drop('text', axis=1, inplace=False).values.tolist()
        emotions_augs = df_augs.drop('text', axis=1, inplace=False).values.tolist()

        
        if not os.path.exists('data/Emotions/classes_df.parquet'): save_file(classes, 'data/Emotions/classes_df.parquet')
        if not os.path.exists('data/Emotions/x_df.parquet'): save_file(sentences, 'data/Emotions/x_df.parquet')
        if not os.path.exists('data/Emotions/y_df.parquet'): save_file(emotions, 'data/Emotions/y_df.parquet')
        if augment:
            if not os.path.exists('data/Emotions/y_augs_df.parquet'): save_file(emotions_augs, 'data/Emotions/y_augs_df.parquet')
            if not os.path.exists('data/Emotions/x_augs_df.parquet'): save_file(sentences_augs, 'data/Emotions/x_augs_df.parquet')
        

    return sentences, emotions, sentences_augs, emotions_augs, classes

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

    if False:
        # slang datasets
        download_dataset(
                'https://www.kaggle.com/datasets/rtatman/spelling-variation-on-urban-dictionary', 
                kaggle_api_key, 
                files_to_move={'spelling_variants_valid.csv': 'Slang/spelling_variants_valid.csv'},
                delete=True,
                dest_name='Slang',
                min_files_expected=1
                )

        download_dataset(
                'https://www.kaggle.com/datasets/gowrishankarp/chat-slang-abbreviations-acronyms', 
                kaggle_api_key, 
                files_to_move={'slang.csv': 'Slang/slang.csv'},
                delete=True,
                dest_name='Slang',
                min_files_expected=2
                )

        download_dataset(
                'https://www.kaggle.com/datasets/rizdelhi/socialmediaabbrevations', 
                kaggle_api_key, 
                files_to_move={'abbrevations.csv': 'Slang/abbrevations.csv'},
                delete=True,
                dest_name='Slang',
                min_files_expected=3
                )

        download_dataset(
                'https://www.kaggle.com/datasets/gogylogy/twitterslang', 
                kaggle_api_key, 
                files_to_move={'twitterSlang.csv': 'Slang/twitterSlang.csv'},
                delete=True,
                dest_name='Slang',
                min_files_expected=4
                )
    
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
    """Returns the sentence vectors for the given sentences.
    
    Args:
        sentences (list): A list of sentences.
        
    Returns:
        torch.Tensor: A tensor of shape (len(sentences), 768) containing the sentence vectors."""
    return [tokenizer.encode(preprocess(sentence), add_special_tokens=True, max_length=128, truncation=True, padding="max_length") for sentence in sentences]

def download_file(url, filename, move_to=None):
    """Downloads a file from a url and saves it to the specified filename. Not used in the current version but useful on windows, where this sometimes is needed as an alternative to wget.
    
    Args:
        url (str): The url to download the file from.
        filename (str): The filename to save the file to.
    """
    if not os.path.exists(move_to + filename) and not os.path.exists(move_to + filename.replace(filename.split('.')[-1], 'parquet')):
        # create the directory if it doesn't exist
        
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            data = response.read() # a `bytes` object
            out_file.write(data)

    if move_to is not None:
        if not os.path.exists(os.path.dirname(move_to)):
            os.makedirs(os.path.dirname(move_to), exist_ok=True)
        if not os.path.exists(move_to + filename) and not os.path.exists(move_to + filename.replace(filename.split('.')[-1], 'parquet')):
            shutil.move(filename, move_to)