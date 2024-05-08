import pandas as pd
import numpy as np
from itertools import combinations
from torch import optim, nn, utils, from_numpy, zeros, float64, argmax
from torchmetrics import Accuracy
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar

BATCH_SIZE = 64

class ViralKineticsDNN(L.LightningModule):
    def __init__(self, in_features, num_buckets):
        super().__init__()
        self.in_features = set(in_features)
        self.num_buckets = num_buckets
        self.stack = nn.Sequential(
            nn.Linear(len(self.in_features), 2 * len(self.in_features) * num_buckets, dtype=float64),
            nn.ReLU(),
            nn.Linear(2 * len(self.in_features) * num_buckets, len(self.in_features) * num_buckets, dtype=float64),
            nn.ReLU(),
            nn.Linear(len(self.in_features) * num_buckets, num_buckets, dtype=float64)
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_buckets)

    def forward(self, x):
        return self.softmax(self.stack(x).squeeze())

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
        prediction = argmax(self.softmax(result), dim=1)
        accuracy = self.accuracy(prediction, argmax(y, dim=1).long().flatten())
        self.log("validation_loss", loss)
        self.log("validation_accuracy", accuracy)


    def test_step(self, batch, batch_idx):
        x, y = batch
        result = self.stack(x)
        loss = self.loss_function(result, y)
        prediction = argmax(self.softmax(result), dim=1)
        accuracy = self.accuracy(prediction, argmax(y, dim=1).long().flatten())
        self.log("testing_loss", loss)
        self.log("testing_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class DDEDataset(utils.data.Dataset):
    def __init__(self, path, atr, has_noise, input_features, num_buckets):
        data = pd.read_csv(path)
        
        if has_noise:
            noise = np.random.normal(0, 10000, [len(data), 12])
            data = data + noise

        data = data.mask(data < 1, 1)
        data[['xTarget', 'xPre-Infected', 'xInfected', 'xVirus', 'xCDE8e', 'xCD8m']] = np.log(data[['xTarget', 'xPre-Infected', 'xInfected', 'xVirus', 'xCDE8e', 'xCD8m']])
        x_cols = ['xTarget', 'xPre-Infected', 'xInfected', 'xVirus', 'xCDE8e', 'xCD8m']
        y_cols = ['yTarget', 'yPre-Infected', 'yInfected', 'yVirus', 'yCDE8e', 'yCD8m']

        removed_x_cols = []
        removed_features = list(set([0,1,2,3,4,5]) - set(input_features))
        for feature in removed_features:
            removed_x_cols.append(x_cols[feature])
        data = data.drop(columns=removed_x_cols)

        y_cols.pop(atr)
        data = data.drop(columns=y_cols)

        self.atr = atr
        self.num_buckets = num_buckets
        self.y_max = data.max(axis=0).iloc[-1]
        self.y_min = data.min(axis=0).iloc[-1]
        self.data = data

    def __len__(self):
        return len(self.data)

    def bracket(self, y):
        bucket_size = (self.y_min + self.y_max) / self.num_buckets
        current_location = self.y_min + bucket_size
        bucket = 0
        while current_location < y:
            current_location += bucket_size
            bucket += 1
        return bucket

    def drop_rows(self, rows):
        self.data = self.data.drop(rows).reset_index(drop=True)
        self.y_max = self.data.max(axis=0).iloc[-1]
        self.y_min = self.data.min(axis=0).iloc[-1]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = from_numpy(row[0:len(row) - 1].to_numpy()).double()
        y = row.iloc[-1]
        y_tensor = zeros(self.num_buckets).double()
        y_tensor[self.bracket(y)] = 1
        return x, y_tensor

def make_dataset(training_dataset_path, testing_dataset_path, input_features, output_feature, has_noise=False, num_buckets=4, dataset_usage_steps=0):
    dataset = DDEDataset(training_dataset_path, output_feature, has_noise, input_features, num_buckets)
    
    for i in range(dataset_usage_steps):
        rows_to_drop = []
        for j in range(len(dataset)):
            if j % 2 != 0:
                rows_to_drop.append(j)
        dataset.drop_rows(rows_to_drop)

    training_set, validation_set = utils.data.random_split(dataset, [.8, .2])
    testing_set = DDEDataset(testing_dataset_path, output_feature, False, input_features, num_buckets)
    return (training_set, validation_set, testing_set)

def run_training(model, training_set, validation_set, testing_set, version=0, epochs=100, model_name=None):
    training_loader = utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    validation_loader = utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True, pin_memory=True)
    testing_loader = utils.data.DataLoader(testing_set, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True, pin_memory=True)
    trainer = L.Trainer(max_epochs=epochs, check_val_every_n_epoch=10, accelerator='auto', log_every_n_steps=2, logger=TensorBoardLogger("lightning_logs", name=model_name, version=version), callbacks=[EarlyStopping("validation_loss", min_delta=0.001), RichProgressBar()])
    trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=validation_loader)
    return trainer.validate(model, dataloaders=validation_loader), trainer.test(model, dataloaders=testing_loader)

def testing_average(num_tests, input_features, num_buckets, training_dataset_path, testing_dataset_path, output_feature, has_noise=False, dataset_usage_steps=0):
    total_val_loss = 0
    total_val_accuracy = 0
    total_loss = 0
    total_accuracy = 0
    for i in range(num_tests):
        model = ViralKineticsDNN(input_features, num_buckets)
        training_set, validation_set, testing_set = make_dataset(training_dataset_path, testing_dataset_path, input_features, output_feature, has_noise=has_noise, num_buckets=num_buckets, dataset_usage_steps=dataset_usage_steps)
        validation_results, testing_results = run_training(model, training_set, validation_set, testing_set, version=i, epochs=10000, model_name="ViralKineticsDDE_" + str(input_features) + "_" + str(output_feature) + "_" + str(dataset_usage_steps) + "_" + str(num_buckets))
        total_val_loss += validation_results[0]['validation_loss']
        total_val_accuracy += validation_results[0]['validation_accuracy']
        total_loss += testing_results[0]["testing_loss"]
        total_accuracy += testing_results[0]["testing_accuracy"]
    return (total_val_loss/num_tests, total_val_accuracy/num_tests, total_loss/num_tests, total_accuracy/num_tests)

def perform_experiment(output_path, training_path, testing_path, data_usage_steps, initial_num_buckets, final_num_buckets, buckets_multiplier, input_combinations, output_features, num_tests_per_model=3, has_noise=False):
    final_results = []
    for steps in range(data_usage_steps):
        num_buckets = initial_num_buckets
        while (num_buckets <= final_num_buckets):
            for combination in input_combinations:
                for output_feature in output_features:
                    results = testing_average(num_tests_per_model, combination, num_buckets, training_path, testing_path, output_feature, dataset_usage_steps=steps, has_noise=has_noise)
                    final_results.append((combination, output_feature, 100/np.power(2, steps), num_buckets, results[0], results[1], results[2], results[3]))
            num_buckets *= buckets_multiplier
    final_results = pd.DataFrame(final_results, columns=["input_features", "output_feature", "data_usage", "num_buckets", "average_final_validation_loss", "average_final_validation_accuracy", "average_testing_loss", "average_testing_accuracy"])
    final_results.to_csv(output_path)

if __name__ == '__main__':
   perform_experiment("results/immune_memory_results.csv", "data/viral_kinetics_none_0.001_1_0_12.csv", "data/viral_kinetics_beta_delta_e_1_0.001_1_0_12.csv", 8, 4, 16, 2, list(combinations((0,1,2,3,4,5), 5)) + [(0,1,2,3,4,5)], [2,3,4,5], has_noise=True)