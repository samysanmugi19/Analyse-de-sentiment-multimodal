# 1️⃣ Imports
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 2️⃣ Classe BertTextEncoder (pour obtenir la séquence brute de BERT)
class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()
        self.use_finetune = use_finetune
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertModel.from_pretrained(pretrained)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text_tensor):
        """
        text_tensor: (batch_size, 3, seq_len)
        3: input_ids, attention_mask, token_type_ids
        """
        input_ids = text_tensor[:,0,:].long()
        attention_mask = text_tensor[:,1,:].float()
        token_type_ids = text_tensor[:,2,:].long()

        if self.use_finetune:
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )[0]

        return last_hidden_states  # (batch_size, seq_len, hidden_dim)

# 3️⃣ Classe TextEncoder (ajoute $E_m$, Transformer, garde T=8 tokens)
class TextEncoder(nn.Module):
    def __init__(self, hidden_dim=768, T=8, nhead=8, num_layers=1):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim

        # Token spécial E_m
        self.E_m = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, bert_seq):
        """
        bert_seq: (batch_size, seq_len, hidden_dim)
        """
        batch_size = bert_seq.size(0)

        # Ajouter E_m au début
        E_m_expand = self.E_m.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_dim)
        seq = torch.cat([E_m_expand, bert_seq], dim=1)     # (batch_size, seq_len+1, hidden_dim)

        # Transformer attend (seq_len+1, batch_size, hidden_dim)
        seq = seq.permute(1, 0, 2)
        seq = self.transformer(seq)
        seq = seq.permute(1, 0, 2)  # Retour à (batch_size, seq_len+1, hidden_dim)

        # Garder les T premiers tokens
        X_t = seq[:, :self.T, :]
        return X_t  # (batch_size, T, hidden_dim)

