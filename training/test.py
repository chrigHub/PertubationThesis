import pickle

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
import datetime
import os
import re
import ast
import importlib
import sys
import json

sys.path.insert(0, './..')
from utils import data_manage_utils, train_utils

importlib.reload(data_manage_utils)
importlib.reload(train_utils)



clf = RandomForestClassifier()
pickle.dump(clf)


