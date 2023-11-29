import pickle

from sklearn.ensemble import RandomForestClassifier
import importlib
import sys

sys.path.insert(0, '../..')
from main.utils import train_utils, data_manage_utils

importlib.reload(data_manage_utils)
importlib.reload(train_utils)



clf = RandomForestClassifier()
pickle.dump(clf)


