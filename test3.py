import torch
import torchtext
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.functional import one_hot
import torch.nn as nn
import torch.optim as optim

#read dataset
with open("tt.txt") as file:
    text = file.read()

tokenizer = get_tokenizer('basic_english')

tokenized_titles = [tokenizer(title) for title in text]

print(tokenized_titles)