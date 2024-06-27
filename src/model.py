import torch.nn as nn
import torch
from config import *
from torchcrf import CRF

class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)  # Adjusted for the new 'EXCLAMATION'
        self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        
        # Freeze BERT layers if specified
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=self.output_dim)  # Updated for 'EXCLAMATION'

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        
        # BERT encoding
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        x = torch.transpose(x, 0, 1)  # (B, N, E) -> (N, B, E)
        
        # LSTM layer
        x, (_, _) = self.lstm(x)
        x = torch.transpose(x, 0, 1)  # (N, B, E) -> (B, N, E)
        
        # Linear layer for punctuation prediction
        x = self.linear(x)
        return x


class DeepPunctuationCRF(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuationCRF, self).__init__()
        self.bert_lstm = DeepPunctuation(pretrained_model, freeze_bert, lstm_dim)
        self.output_dim = len(punctuation_dict) + 1  # CRF output dimension including 'EXCLAMATION'
        self.crf = CRF(self.output_dim, batch_first=True)  # CRF with updated output dimension

    def log_likelihood(self, x, attn_masks, y):
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        
        x = self.bert_lstm(x, attn_masks)
        attn_masks = attn_masks.byte()
        
        # CRF decoding
        dec_out = self.crf.decode(x, mask=attn_masks)
        y_pred = torch.zeros(y.shape).long().to(y.device)
        
        # Mapping decoded indices to y_pred tensor
        for i in range(len(dec_out)):
            y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        
        return y_pred
