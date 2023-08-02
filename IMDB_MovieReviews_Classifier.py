from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 400
EPOCHS = 10
RANDOM_SEED = 42
BATCH_SIZE = 16
LOAD_MODEL = False


class MovieReviewDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return self.reviews.shape[0]
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
      truncation = True
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    input_ids = torch.Tensor(input_ids)
    attention_mask = torch.Tensor(attention_mask)
    
    outputs = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_output = outputs[1]
    output = self.drop(pooled_output)
    return self.out(output)
  


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = MovieReviewDataset(
    reviews=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  # torch.utils.data.DataLoader
  return DataLoader(
    dataset=ds,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True
  )




def to_sentiment(rating):
  rating = str(rating)
  if rating == 'positive':
    return 0
  else: 
    return 1



def make_dataset():
  df = pd.read_csv('./data/IMDB Dataset.csv')
  df['sentiment'] = df.sentiment.apply(to_sentiment)
  df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
  df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
  train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
  val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
  test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

  return df_train, df_val, df_test, train_data_loader, val_data_loader, test_data_loader



def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  with tqdm(data_loader, unit="batch") as tepoch:
    for batch_data in tepoch:
      input_ids = batch_data["input_ids"].to(device)
      attention_mask = batch_data["attention_mask"].to(device)
      targets = batch_data["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      torch.cuda.empty_cache()

  return correct_predictions.double() / n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
      torch.cuda.empty_cache()

  return correct_predictions.double() / n_examples, np.mean(losses)



def model_train(model):
  df_train, df_val, _, train_data_loader, val_data_loader, _ = make_dataset()

  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )

  loss_fn = nn.CrossEntropyLoss().to(device)

  history = defaultdict(list)
  best_accuracy = 0

  for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,    
      loss_fn, 
      optimizer, 
      device, 
      scheduler, 
      len(df_train)
    )

    print(f'Train loss {train_loss} / accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn, 
      device, 
      len(df_val)
    )

    print(f'Val   loss {val_loss} / accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), './model_states/best_model_state.bin')
      best_accuracy = val_acc

    torch.save(model.state_dict(), './model_states/last_model_state.bin')



if __name__ == '__main__':
  # sns.set(style='whitegrid', palette='muted', font_scale=1.2)
  # HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
  # sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
  # rcParams['figure.figsize'] = 12, 8

  np.random.seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)

  class_names = ['negative', 'positive']
  model = SentimentClassifier(len(class_names))
  if LOAD_MODEL == True:
    model.load_state_dict(torch.load("./model_states/best_model_state.bin"), strict=False)
  model = model.to(device)

  model_train(model)
