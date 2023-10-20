import pandas as pd
import numpy as np
import torch.utils.data
from torch import optim, nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
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
            nn.Linear(6, 64, dtype=torch.float64).to(DEVICE),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(64, 124, dtype=torch.float64).to(DEVICE),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(124, 256, dtype=torch.float64).to(DEVICE),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(256, 512, dtype=torch.float64).to(DEVICE),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(512, 1024, dtype=torch.float64).to(DEVICE),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(1024, 2048, dtype=torch.float64).to(DEVICE),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(2048, 2548, dtype=torch.float64).to(DEVICE),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(2548, 6, dtype=torch.float64).to(DEVICE),
        )
        self.loss_function = nn.functional.mse_loss

    def forward(self, x):
        return self.stack(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        result = self.stack(x)
        loss = self.loss_function(result, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.stack(x)
        loss = self.loss_function(result, y)
        self.log("testing_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
        return {'optimizer': optimizer, 'lr_scheduler': {"scheduler": scheduler}}


class ODEDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        # self.data = self.data.mask(self.data < 0, 0)
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
        x = torch.from_numpy(row[0:6].to_numpy()).double().to(DEVICE)
        y = torch.from_numpy(row[6:12].to_numpy()).double().to(DEVICE)
        return x, y


if __name__ == '__main__':
    dataset0 = ODEDataset("data/viral_kinetics_none_0.01_0_12.csv")
    dataset1 = ODEDataset("data/viral_kinetics_none_0.01_0_10.csv")
    dataset2 = ODEDataset("data/viral_kinetics_none_0.01_0_8.csv")
    dataset3 = ODEDataset("data/viral_kinetics_none_0.01_0_6.csv")
    dataset4 = ODEDataset("data/viral_kinetics_none_0.01_2_12.csv")
    dataset5 = ODEDataset("data/viral_kinetics_none_0.01_2_10.csv")
    dataset6 = ODEDataset("data/viral_kinetics_none_0.01_2_8.csv")
    dataset7 = ODEDataset("data/viral_kinetics_none_0.01_2_6.csv")

    training_dataset0, testing_dataset0 = torch.utils.data.random_split(dataset0, [.85, .15])
    training_dataset1, testing_dataset1 = torch.utils.data.random_split(dataset1, [.85, .15])
    training_dataset2, testing_dataset2 = torch.utils.data.random_split(dataset2, [.85, .15])
    training_dataset3, testing_dataset3 = torch.utils.data.random_split(dataset3, [.85, .15])
    training_dataset4, testing_dataset4 = torch.utils.data.random_split(dataset4, [.85, .15])
    training_dataset5, testing_dataset5 = torch.utils.data.random_split(dataset5, [.85, .15])
    training_dataset6, testing_dataset6 = torch.utils.data.random_split(dataset6, [.85, .15])
    training_dataset7, testing_dataset7 = torch.utils.data.random_split(dataset7, [.85, .15])

    training_loader0 = torch.utils.data.DataLoader(training_dataset0, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader0 = torch.utils.data.DataLoader(testing_dataset0, batch_size=BATCH_SIZE)
    training_loader1 = torch.utils.data.DataLoader(training_dataset1, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader1 = torch.utils.data.DataLoader(testing_dataset1, batch_size=BATCH_SIZE)
    training_loader2 = torch.utils.data.DataLoader(training_dataset2, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader2 = torch.utils.data.DataLoader(testing_dataset2, batch_size=BATCH_SIZE)
    training_loader3 = torch.utils.data.DataLoader(training_dataset3, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader3 = torch.utils.data.DataLoader(testing_dataset3, batch_size=BATCH_SIZE)
    training_loader4 = torch.utils.data.DataLoader(training_dataset4, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader4 = torch.utils.data.DataLoader(testing_dataset4, batch_size=BATCH_SIZE)
    training_loader5 = torch.utils.data.DataLoader(training_dataset5, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader5 = torch.utils.data.DataLoader(testing_dataset5, batch_size=BATCH_SIZE)
    training_loader6 = torch.utils.data.DataLoader(training_dataset6, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader6 = torch.utils.data.DataLoader(testing_dataset6, batch_size=BATCH_SIZE)
    training_loader7 = torch.utils.data.DataLoader(training_dataset7, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader7 = torch.utils.data.DataLoader(testing_dataset7, batch_size=BATCH_SIZE)

    model0 = ViralKineticsDNN()
    model1 = ViralKineticsDNN()
    model2 = ViralKineticsDNN()
    model3 = ViralKineticsDNN()
    model4 = ViralKineticsDNN()
    model5 = ViralKineticsDNN()
    model6 = ViralKineticsDNN()
    model7 = ViralKineticsDNN()

    trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-0 to 12 Model", version="version_2"))
    trainer.fit(model=model0, train_dataloaders=training_loader0, val_dataloaders=testing_loader0)
    trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-0 to 10 Model", version="version_2"))
    trainer.fit(model=model1, train_dataloaders=training_loader1, val_dataloaders=testing_loader1)
    trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-0 to 8 Model", version="version_2"))
    trainer.fit(model=model2, train_dataloaders=training_loader2, val_dataloaders=testing_loader2)
    trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-0 to 6 Model", version="version_2"))
    trainer.fit(model=model3, train_dataloaders=training_loader3, val_dataloaders=testing_loader3)
    trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-2 to 12 Model", version="version_2"))
    trainer.fit(model=model4, train_dataloaders=training_loader4, val_dataloaders=testing_loader4)
    trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-2 to 10 Model", version="version_2"))
    trainer.fit(model=model5, train_dataloaders=training_loader5, val_dataloaders=testing_loader5)
    trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-2 to 8 Model", version="version_2"))
    trainer.fit(model=model6, train_dataloaders=training_loader6, val_dataloaders=testing_loader6)
    rainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, logger=TensorBoardLogger("lightning_logs", name="0.01-2 to 6 Model", version="version_2"))
    trainer.fit(model=model7, train_dataloaders=training_loader7, val_dataloaders=testing_loader7)

    # dataset1 = ODEDataset("data/viral_kinetics_none_0.01_0_12.csv")
    # dataset2 = ODEDataset("data/viral_kinetics_none_0.001_0_12.csv")
    # dataset3 = ODEDataset("data/viral_kinetics_none_0.0001_0_12.csv")

    # training_dataset1, testing_dataset1 = torch.utils.data.random_split(dataset1, [.85, .15])
    # training_dataset2, testing_dataset2 = torch.utils.data.random_split(dataset2, [.85, .15])
    # training_dataset3, testing_dataset3 = torch.utils.data.random_split(dataset3, [.85, .15])
    #
    # training_loader1 = torch.utils.data.DataLoader(training_dataset1, batch_size=BATCH_SIZE, shuffle=True)
    # testing_loader1 = torch.utils.data.DataLoader(testing_dataset1, batch_size=BATCH_SIZE)
    # training_loader2 = torch.utils.data.DataLoader(training_dataset2, batch_size=BATCH_SIZE, shuffle=True)
    # testing_loader2 = torch.utils.data.DataLoader(testing_dataset2, batch_size=BATCH_SIZE)
    # training_loader3 = torch.utils.data.DataLoader(training_dataset3, batch_size=BATCH_SIZE, shuffle=True)
    # testing_loader3 = torch.utils.data.DataLoader(testing_dataset3, batch_size=BATCH_SIZE)
    #
    # model1 = ViralKineticsDNN()
    # model2 = ViralKineticsDNN()
    # model3 = ViralKineticsDNN()

    #trainer = pl.Trainer(log_every_n_steps=1, accelerator=DEVICE, max_epochs=10000, callbacks=[checkpoint_callback], logger=TensorBoardLogger("lightning_logs", name="0.01 to 0.01 Model", version="version_1"))
    #trainer.fit(model=model1, train_dataloaders=training_loader1, val_dataloaders=testing_loader1)
    #trainer = pl.Trainer(log_every_n_steps=10, accelerator=DEVICE, max_epochs=1000, callbacks=[checkpoint_callback], logger=TensorBoardLogger("lightning_logs", name="0.001 to 0.001 Model", version="version_1"))
    #trainer.fit(model=model2, train_dataloaders=training_loader2, val_dataloaders=testing_loader2)
    #trainer = pl.Trainer(log_every_n_steps=100, accelerator=DEVICE, max_epochs=100, callbacks=[checkpoint_callback], logger=TensorBoardLogger("lightning_logs", name="0.0001 to 0.0001 Model", version="version_1"))
    #trainer.fit(model=model3, train_dataloaders=training_loader3, val_dataloaders=testing_loader3)
