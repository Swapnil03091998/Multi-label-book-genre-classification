import pandas as pd
import numpy as np
import os
import re
import string
import json
import emoji
import numpy as np
import pandas as pd
from sklearn import metrics
from bs4 import BeautifulSoup
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer, BertModel, BertConfig, AutoModel, AdamW
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("/kaggle/input/best-books-10k-multi-genre-data/goodreads_data.csv").drop("Unnamed: 0", axis=1)

# Preprocess the data
data = data.dropna()
data = data.drop_duplicates()

#Extract all the distinct genres from the data
genre_set = set()
for s in data['Genres']:
    # Extracting genres from the string
    genres = [genre.strip(" '") for genre in s.strip("[]").split(",")]

    # Creating a set of genres
    genres = set(genres)
    genre_set.update(genres)

genre_set.remove('')

print(len(genre_set)) #gives 617

#Getting count of each genre in the data
genre_list = list()
for s in data['Genres']:
    # Extracting genres from the string
    genres = [genre.strip(" '") for genre in s.strip("[]").split(",")]

    genre_list.extend(genres)
    
from collections import Counter
genres_count = dict(Counter(genre_list))

genres_count_sorted = {k: v for k, v in sorted(genres_count.items(), key=lambda item: item[1],
                                              reverse=True)}

#checking if there are any repetitions of any genre in upper and lower cases
len({x.lower() for x in genre_set})

import statistics
counts = list(genres_count.values())
print(max(counts))
print(min(counts))
print(sum(counts)/len(counts))
print(statistics.median(counts))

#average frequency of each genre is around 95
top_genres = {k:v for k, v in genres_count.items() if v>=95}
print(len(top_genres)) #gives 100

top_genres = list(genres_count_sorted.keys())[:5] #taking top 5 genres for avoiding sparsity problem during model training

print(top_genres) #['Fiction', 'Nonfiction', 'Fantasy', 'Classics', 'Romance']

df = data.copy()
df = df[['Description', 'Genres']]

#['Fiction', 'Nonfiction', 'Fantasy', 'Classics', 'Romance']

is_fiction = list()
is_nonfiction = list()
is_fantasy = list()
is_classics = list()
is_romance = list()

def make_genre_map_list(genres, listname, genrename):
    if genrename in genres:
        listname.append(1)
    else:
        listname.append(0)

for s in df['Genres']:
    # Extracting genres from the string
    genres = [genre.strip(" '") for genre in s.strip("[]").split(",")]
    
    make_genre_map_list(genres, is_fiction, 'Fiction')
    make_genre_map_list(genres, is_nonfiction, 'Nonfiction')
    make_genre_map_list(genres, is_fantasy, 'Fantasy')
    make_genre_map_list(genres, is_classics, 'Classics')
    make_genre_map_list(genres, is_romance, 'Romance')

df['is_fiction'] = is_fiction
df['is_nonfiction'] = is_nonfiction 
df['is_fantasy'] = is_fantasy 
df['is_classics'] = is_classics
df['is_romance'] = is_romance

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 200
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 2e-5
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

target_cols = df.drop(['Description', 'Genres'], axis=1).columns

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.text = df.Description
        self.tokenizer = tokenizer
        self.targets = df[target_cols].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
      
df = df.reset_index(drop=True)
df_train = df.sample(frac=0.85)
df_valid = df.drop(df_train.index)
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

train_dataset = BERTDataset(df_train, tokenizer, MAX_LEN)
valid_dataset = BERTDataset(df_valid, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, 
                          num_workers=4, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, 
                          num_workers=4, shuffle=False, pin_memory=True)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base')
#         self.l2 = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768,5)
    
    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
#         output_2 = self.l2(output_1)
        output = self.fc(features)
        return output

model = BERTClass()
model.to(device);

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
  
optimizer = AdamW(params =  model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
def train(epoch):
    model.train()
    for _,data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)
        if _%500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
for epoch in range(EPOCHS):
    train(epoch)
    
#At the end of the 10th epoch, the loss was around 0.06

def validation():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(valid_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
  
outputs, targets = validation()
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")

# Accuracy Score = 0.5940860215053764
# F1 Score (Micro) = 0.8126036484245438
# F1 Score (Macro) = 0.7847748988746912

torch.save(model.state_dict(), 'model.bin')


