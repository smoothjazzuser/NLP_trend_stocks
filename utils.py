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
import time
import datetime
import re
from getpass import getpass
from shutil import rmtree
import os
pd.set_option('io.parquet.engine', 'pyarrow')
import fasttext
import fasttext.util

def download_datasets(url:str, unzip:bool=True, delete_zip:bool=True, files_to_move:dict = {}, delete=False, dest_name:str = None, verbose:bool = True):
    """Downloads the datasets from kaggle using the official kaggle api.
    
    See this forumn for more information, as the official documentation is lacking:
        https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python"""
    from kaggle.api.kaggle_api_extended import KaggleApi 

    api = KaggleApi()
    api.authenticate()

    if "datasets" in url: url = url.split("datasets/")[-1]
    if not os.path.exists('data/'): os.mkdir('data/')
    if not os.path.exists('data/{}'.format(dest_name)): os.mkdir('data/{}'.format(dest_name))
    if delete_zip:
        for file in glob('data/*.zip'):
            os.remove(file)
        
    if not list(files_to_move.values())[0].split('/')[-1].split('.')[0] in [file.split('\\')[-1].split('.')[0] for file in glob('data/*/*')] +  [file.split('\\')[-1].split('.')[0] for file in glob('data/*/*/*')] + [file.split('\\')[-1].split('.')[0] for file in glob('data/*')]:

        api.dataset_download_files(url, path='data/', unzip=unzip, quiet=not verbose)

        for k, v in files_to_move.items():
            if os.path.exists(f'data/{k}'):
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
        os.removedirs('data/Stocks')
        os.rename('data/Data/Stocks', 'data/Stocks')
        rmtree('data/Data')

    # convert the files to parquet format, which is a much better for this project
    csv_files_to_compress = glob('data/*/*.csv') + glob('data/*/*/*.csv') + glob('data/*.csv')
    for file in csv_files_to_compress:
        if not os.path.exists(file.replace('.csv', '.parquet')):
            df = load_file(file)
            df.to_parquet(file.replace('.csv', '.parquet'), compression='brotli', engine='pyarrow')
        if os.path.exists(file):
            os.remove(file)

    xlsx_files_to_compress = glob('data/*/*.xlsx') + glob('data/*/*/*.xlsx') + glob('data/*.xlsx')
    for file in xlsx_files_to_compress:
        if not os.path.exists(file.replace('.xlsx', '.parquet')):
            df = load_file(file)
            df.to_parquet(file.replace('.xlsx', '.parquet'), compression='brotli', engine='pyarrow')
        if os.path.exists(file):
            os.remove(file)

def load_file(file:str):
    """Reads file into a pandas dataframe. Supports csv, parquet and xlsx files. """
    file_type = file.split('.')[-1]
    if file_type == 'parquet':
        return pd.read_parquet(file)
    elif file_type == 'csv':
        return pd.read_csv(file, low_memory=False, parse_dates=True, infer_datetime_format=True, on_bad_lines='skip', encoding_errors= 'replace')
    elif file_type == 'xlsx':
        return pd.read_excel(file, parse_dates=True)
    elif file_type == 'pkl':
        return load(file, compression='lz4')
    elif file_type == '.lz4':
        return load(file, compression='lz4')
    else:
        try:
            return load(file)
        except:
            raise ValueError("File type not supported.")

def save_file(df, file:str):
    """Saves a pandas dataframe to a file."""
    file_type = file.split('.')[-1]
    if file_type == 'parquet':
        df.to_parquet(file, compression='brotli', engine='pyarrow')
    elif file_type == 'pkl':
        dump(df, file + '.lz4', compression='lz4')
    elif file_type == 'csv':
        df.to_csv(file, index=False)
    elif file_type == 'xlsx':
        df.to_excel(file, index=False)

def fernet_key_encryption(password:str, name:str):
    """Encrypts and decrypts a key using Fernet encryption. 
    
    Args:
    
        password (str): The password to encrypt the key with.
        
        name (str): The name of the key to encrypt.
        
    Returns:
    Saves the encrypted key to a file and returns the decrypted key."""
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

def cpi_adjust(df: pd.DataFrame, cpi: pd.DataFrame):
    """Adjusts the dataframe to account for inflation."""

    for col in [x for x in df.columns if x not in ["CPI", "date", "Unemp_rate", "mortgage_rate"]]:
        df[col] = df[col].div(cpi.CPI, axis=0)

    return df

def parse_emotion_dataframes(selection: int = [0, 1, 2, 3, 4], ensure_only_one_label: bool = True):
    """Parses the emotion dataframes. 
    
    Args: selection (int): The emotion dataframes to parse. 0 = GoEmotions, 1 = """

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

    return df

def interpolate_months_to_days(df: pd.DataFrame, extend_trend_to_today: bool = False):
    """Interpolates the dataframe to account for missing days. E.g, Macro data is usually available on a monthly basis.
    
    Args: DataFrame, extend_trend_to_today (bool): If True, the index will be extended to today (only enabled for monthly data).)"""
    #check if index is already datetime
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
        df.index = pd.to_datetime(df[col])
        df.drop(columns=[col], inplace=True)

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

def intersect_df(df1: pd.DataFrame, df2: pd.DataFrame, interpolate_to_days: bool = False, extend_trend_to_today: bool = False):
    """Performantly Intersects two dataframes based on their index.
    
    Args: Two dataframes with a datetime index.
    
    Returns: The datapoints shared between the two datasets."""

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
    """A work-in-progress. Goal is to take stock ticker symbols and return a list of search terms for NLP web scraping. Officers, affiliated companies, company name, etc."""
    def __init__(self, file_path = 'data/Stocks/', file_ext = '.parquet'):
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
        directions = "\n Please ensure that you have downloaded the stock data first from https://www.kaggle.com/datasets/footballjoe789/us-stock-dataset. \n Then extract Stocks/ to the data/ folder and move Stock_List to data/. \n Then run this script again. \n Thank you. \n"

        if os.path.exists("data/Stock_List.parquet") == False:
            error = "data/Stock_List.parquet is missing. "

        if os.path.exists("data/Stocks/"):
            files_found = len(glob("data/Stocks/*"))
            if files_found == 0:
                error = "No stock data found in data/Stocks/ folder. "
        else:
            error = "data/Stocks/ folder is missing. "
        
        if error != "":
            print(error + directions)
            self.data = pd.DataFrame()
            self.stocks_symbols = []
            self.yh_tickers = []
        return True if error == "" else False

def get_emotion_df():
    if os.path.exists('data/Emotions/emotion_df.parquet'):
        emotion_df = load_file('data/Emotions/emotion_df.parquet')
    else:
        emotion_df = parse_emotion_dataframes([0, 1, 2, 3, 4], ensure_only_one_label=True)
        if not os.path.exists('data/Emotions/cc.en.300.bin'):
            fasttext.util.download_model('en', if_exists='ignore')
            os.rename('cc.en.300.bin', 'data/Emotions/cc.en.300.bin')
            os.remove('cc.en.300.bin.gz')
            fast_text_model = fasttext.load_model('data/Emotions/cc.en.300.bin')
        else:
            fast_text_model = fasttext.load_model('data/Emotions/cc.en.300.bin')
        # preprocess the text data using the fasttext model
        emotion_df['text'] = emotion_df['text'].apply(lambda x: fast_text_model.get_sentence_vector(x))
        # save the preprocessed data to a file as a parquet file
        save_file(emotion_df, 'data/Emotions/emotion_df.parquet')

    return emotion_df

