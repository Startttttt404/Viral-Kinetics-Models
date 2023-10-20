import pandas as pd
import numpy as np
import torch.utils.data
from torch import optim, nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class ViralKineticsLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(64, 6)
        self.loss_function = nn.functional.mse_loss

    def forward(self, x):
        return self.linear(self.lstm(x)[0])

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.linear(self.lstm(x)[0])
        loss = self.loss_function(output, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.linear(self.lstm(x)[0])
        loss = self.loss_function(output, y)
        self.log("testing_loss", loss)

