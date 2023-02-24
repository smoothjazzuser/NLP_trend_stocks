import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from compress_pickle import dump, load
import os
from yahooquery import Ticker
import timeit
import time
import datetime

class aquire_stock_search_terms():
    """A work-in-progress. Goal is to take stock ticker symbols and return a list of search terms for NLP web scraping. Officers, affiliated companies, company name, etc."""
    def __init__(self, file_path = 'data/Stocks/', file_ext = '.csv'):
        self.file_path = file_path
        self.load_symbols()
        self.ticker_list_to_dataframe()
        self.get_company_list()
        self.process_information()

    def load_symbols(self):
        """Load all the stock symbols."""
        self.stocks_symbols = pd.read_csv("data/Stock_List.csv")['Symbol'].tolist()
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
    
    def process_information(self):
        self.data['Company'] = [self.get_company(t) for t in range(len(self.yh_tickers))]
        self.data["Executive"] = [self.get_company_officers(t) for t in range(len(self.yh_tickers))] 
        self.data['Industry'] = [self.get_industry(t) for t in range(len(self.yh_tickers))]
        self.data['quoteType'] = [self.get_quote_type(t) for t in range(len(self.yh_tickers))]
        self.data['sector'] = [self.get_sector(t) for t in range(len(self.yh_tickers))]


if __name__ == "__main__":
    search_terms = aquire_stock_search_terms()
    search_terms.data.head()