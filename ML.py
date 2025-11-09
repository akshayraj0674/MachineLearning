import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


train_url   = "https://github.com/akshayraj0674/MachineLearning/blob/main/ml-4127-e-project-2/train_data.xlsx"
test_url    = "https://github.com/akshayraj0674/MachineLearning/blob/main/ml-4127-e-project-2/test_data.xlsx"
sample_url  = "https://github.com/akshayraj0674/MachineLearning/blob/main/ml-4127-e-project-2/sample_submission_probs.csv"


TARGET_COL = "action"              # Change if needed
N_FOLDS = 5
RANDOM_STATE = 42

USE_CLASS_WEIGHT = True            # Set False if class distribution is balanced
CALIBRATE = False                  # True => probability calibration layer (usually logistic is already good)
CALIBRATION_METHOD = "isotonic"    # "sigmoid" or "isotonic" (if CALIBRATE=True)
SOLVER = "lbfgs"                   # "lbfgs" stable for multinomial; use "saga" for elastic-net or huge data
MAX_ITER = 3000                    # Increase if convergence warnings appear
VERBOSE = 1