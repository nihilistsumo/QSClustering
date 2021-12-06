import math
import random
import argparse
import transformers
from transformers import AdamW
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from core.clustering import QuerySpecificClusteringModel, SBERTTripletLossModel, get_nmi_loss, \
    get_weighted_adj_rand_loss, QuerySpecificDKM
from util.data import Vital_Wiki_Dataset, get_article_qrels, get_rev_qrels
from tqdm import tqdm, trange
import json
import numpy as np
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)