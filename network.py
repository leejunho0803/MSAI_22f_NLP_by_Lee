import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE>
        self.title_conv = nn.Conv1d(
            in_channels=hid_size,
            out_channels=hid_size,
            kernel_size=2
        )
        self.title_act = nn.ReLU()
        self.title_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE>
        self.full_conv = nn.Conv1d(
            in_channels=hid_size,
            out_channels=hid_size,
            kernel_size=2
        )
        self.full_act = nn.ReLU()
        self.full_pooling = nn.AdaptiveAvgPool1d(1)
        
        # self.category_out = # <YOUR CODE HERE>
        self.category_out = nn.Linear(n_cat_features, hid_size)
        self.category_out_relu = nn.ReLU()

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.inter_dense_relu = nn.ReLU()
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)
        self.final_dense_relu = nn.ReLU()

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        # title = # <YOUR CODE HERE>
        title_relu = self.title_act(title_beg)
        title = self.title_pooling(title_relu)

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        # full = # <YOUR CODE HERE>
        full_relu = self.full_act(full_beg)
        full = self.full_pooling(full_relu)

        category_beg = self.category_out(input3)
        # category = # <YOUR CODE HERE>
        category = self.category_out_relu(category_beg)
        
        print(title.shape)
        print(full.shape)
        print(category.shape)
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        # out = # <YOUR CODE HERE>
        out = self.inter_dense(concatenated)
        out = self.inter_dense_relu(out)
        out = self.final_dense(out)
        out = self.final_dense_relu(out)
        
        return out