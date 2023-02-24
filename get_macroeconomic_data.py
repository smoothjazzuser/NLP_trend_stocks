import pandas as pd
import numpy as np
import seaborn
from glob import glob
from compress_pickle import dump, load
import os
from yahooquery import Ticker
import timeit
import time
import datetime

class get_macroeconomic_data ():
    """Aquire historical macroeconomic data from different sources."""
    def __init__(self, path):
        self.path = path
        self.data = self.get_data()