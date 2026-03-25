import torch
import numpy
import requests
import pandas as pd

from dotenv import load_dotenv
import os 
from datetime import datetime
import time

import matplotlib.pyplot as plt
from torch import nn
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

