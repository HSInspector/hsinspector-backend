import transformers
from transformers import XLNetTokenizer, XLNetModel, AdamW, get_linear_schedule_with_warmup
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from torch import nn, optim
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import XLNetForSequenceClassification
from hate_speech_type import hate_speech_type

class Model():
    def __init__(self) -> None:
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.MAX_LEN=128
        print(self.device)
        
    def load_model(self, path='xlnet-base-cased'):
        self.model = XLNetForSequenceClassification.from_pretrained('./trained_model/xlnet-base-cased', local_files_only = True, num_labels = 5)
        self.model = self.model.to(self.device)
        self.tokenizer = XLNetTokenizer.from_pretrained('./trained_model/xlnet-base-cased/', local_files_only = True)
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def predict_tweet(self, tweet):
    
        review_text = tweet

        encoded_review = self.tokenizer.encode_plus(
        review_text,
        max_length=self.MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = pad_sequences(encoded_review['input_ids'], maxlen=self.MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=self.MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask) 

        input_ids = input_ids.reshape(1,self.MAX_LEN).to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        outputs = outputs[0][0].cpu().detach()

        probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
        _, prediction = torch.max(outputs, dim =-1)

        # print("non hate:", probs[0])
        # print("hate:", probs[1])
        # print("racism:", probs[2])
        # print("sexism:", probs[3])
        # print("islamophobia:", probs[4])
        # print('prediction: ', hate_speech_type(prediction.item()))
        return hate_speech_type(prediction.item())
        # print()
        # print(f' text: {review_text}')
        # print(f'type  : {[prediction]}')
        # print(f'actual  : {[actual]}')


model = Model()
model.load_model('trained_model/xlnet_model4.bin')
model.predict_tweet('islam is good')