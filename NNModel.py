import pandas as pd
import numpy as np
import torch.utils.data
from torch import optim, nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
BATCH_SIZE = 2048

checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="testing_loss",
    mode="min",
    filename="sample-mnist-{epoch:02d}-{testing_loss:.5f}",
)

class ViralKineticsDNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(6, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, 48),
            nn.ReLU(),
            nn.Linear(48, 6),
        )
        self.loss_function = nn.functional.mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        result = self.stack(x)
        loss = self.loss_function(result, y)
        print(result[0])
        print(y[0])
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.stack(x)
        loss = self.loss_function(result, y)
        self.log("testing_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0005)
        return optimizer


class ODEDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.data['xTarget'] = minmax_scale(self.data['xTarget'])
        self.data['xPre-Infected'] = minmax_scale(self.data['xPre-Infected'])
        self.data['xInfected'] = minmax_scale(self.data['xInfected'])
        self.data['xVirus'] = minmax_scale(self.data['xVirus'])
        self.data['xCDE8e'] = minmax_scale(self.data['xCDE8e'])
        self.data['xCD8m'] = minmax_scale(self.data['xCD8m'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.from_numpy(row[0:6].to_numpy()).float()
        y = torch.from_numpy(row[6:12].to_numpy()).float()
        return x, y


if __name__ == '__main__':
    dataset1 = ODEDataset("data/viral_kinetics_none_0.01_0_12.csv")
    dataset2 = ODEDataset("data/viral_kinetics_none_0.001_0_12.csv")
    dataset3 = ODEDataset("data/viral_kinetics_none_0.0001_0_12.csv")

    training_dataset1, testing_dataset1 = torch.utils.data.random_split(dataset1, [.8, .2])
    training_dataset2, testing_dataset2 = torch.utils.data.random_split(dataset2, [.8, .2])
    training_dataset3, testing_dataset3 = torch.utils.data.random_split(dataset3, [.8, .2])

    training_loader1 = torch.utils.data.DataLoader(training_dataset1, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader1 = torch.utils.data.DataLoader(testing_dataset1)
    training_loader2 = torch.utils.data.DataLoader(training_dataset2, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader2 = torch.utils.data.DataLoader(testing_dataset2)
    training_loader3 = torch.utils.data.DataLoader(training_dataset3, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader3 = torch.utils.data.DataLoader(testing_dataset3)

    model1 = ViralKineticsDNN()
    model2 = ViralKineticsDNN()
    model3 = ViralKineticsDNN()

    trainer = pl.Trainer(accelerator=DEVICE, max_epochs=100000, callbacks=[checkpoint_callback])
    trainer.fit(model=model1, train_dataloaders=training_loader1, val_dataloaders=testing_loader1)
    #trainer = pl.Trainer(accelerator=DEVICE, max_epochs=1000, callbacks=[checkpoint_callback])
    #trainer.fit(model=model2, train_dataloaders=training_loader2, val_dataloaders=testing_loader2)
    # trainer = pl.Trainer(accelerator=DEVICE, max_epochs=1000, callbacks=[checkpoint_callback])
    # trainer.fit(model=model3, train_dataloaders=training_loader3, val_dataloaders=testing_loader3)
