import torch
import torch.nn as nn
import torch.nn.functional as F
from read_set import ReadSet
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint


class PEC(pl.LightningModule):
    def __init__(self, input_size=64, num_class=5, lr=1e-02, optimizer=optim.Adam, criterion=nn.NLLLoss()):
        super().__init__()
        self.num_class = num_class
        self.optimizer = optimizer
        self.input_size = input_size
        self.lr = lr
        self.criterion = criterion
        self.accuracy = Accuracy(num_classes=self.num_class)
        self.f1 = F1(num_classes=self.num_class)
        self.recall = Recall(num_classes=self.num_class)
        self.loss = []
        self.acc = []
        self.best_acc = 0
        # self.save_hyperparameters("input_size", "num_class")
        self.m = nn.LogSoftmax(dim=1)

        self.conv1 = nn.Conv1d(input_size, input_size * 4, 3, padding=1)
        self.batch1 = nn.BatchNorm1d(input_size * 4)

        self.conv2 = nn.Conv1d(input_size * 4, input_size * 8, 3, padding=1)
        self.batch2 = nn.BatchNorm1d(input_size * 8)

        self.conv3 = nn.Conv1d(input_size * 8, input_size, 3, padding=1)
        self.batch3 = nn.BatchNorm1d(input_size)

        self.fcl = nn.Linear(input_size * 3, int(input_size / 2))
        self.pool = nn.MaxPool1d(1)
        self.fcl2 = nn.Linear(int(input_size / 2), input_size)
        self.t_conv1 = nn.ConvTranspose1d(input_size, input_size * 4, 2)
        self.t_conv2 = nn.ConvTranspose1d(input_size * 4, input_size * 8, 2)
        self.t_conv3 = nn.ConvTranspose1d(input_size * 8, input_size, 1)
        self.final_conv = nn.Conv1d(input_size, int(input_size / 2), 1)
        self.fc1 = nn.Linear(int(input_size / 2) * 3, int(input_size / 2) * 2)
        self.fc2 = nn.Linear(int(input_size / 2) * 2, int(input_size / 2))
        self.fc3 = nn.Linear(int(input_size / 2), self.num_class)

    def forward(self, x):

        encoder1 = self.pool(self.conv1(x))
        encoder1 = F.relu(self.batch1(encoder1))
        encoder2 = self.pool(self.conv2(encoder1))
        encoder2 = F.relu(self.batch2(encoder2))
        encoder3 = self.pool(self.conv3(encoder2))
        encoder3 = F.relu(self.batch3(encoder3))
        x2 = torch.flatten(encoder3, 1)
        x2 = self.fcl(x2)
        x2 = self.fcl2(x2)
        x2 = x2.view(x2.size(0), self.input_size, 1)
        decoder1 = F.relu(self.t_conv1(x2))
        decoder2 = F.relu(self.t_conv2(decoder1))
        decoder3 = torch.sigmoid(self.t_conv3(decoder2))
        final_conv = self.pool(F.relu(self.final_conv(decoder3)))
        x4 = torch.flatten(final_conv, 1)
        x5 = F.relu(self.fc1(x4))
        x5 = F.relu(self.fc2(x5))
        x6 = self.fc3(x5)
        return x6

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch

        outputs = self.forward(inputs)
        loss = self.criterion(self.m(outputs), labels)
        self.log("train loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch

        outputs = self.forward(inputs)
        loss = self.criterion(self.m(outputs), labels)

        _, predicted = torch.max(outputs, 1)
        acc = self.accuracy(predicted, labels)
        acc_cpu = acc.cpu()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        self.loss.append(loss.item())
        self.acc.append(acc_cpu)
        if acc_cpu > self.best_acc:
            self.best_acc = acc_cpu

    def predict(self, x, device="cpu", class_name=None):
        if class_name is None:
            class_names = ['ape', 'bleach', 'can', 'cat', 'clamp', 'drill', 'duck', 'eggbox', 'glue', 'tomato']
        else:
            class_names = class_name

        # size_data = len(x)
        # mean = np.mean(x, axis=0)
        # x = x - mean

        x = data_tools.fps(x, 64)

        mean = np.mean(x, axis=0)
        x = x - mean

        # sorted_data = input_data[np.argsort(input_data[:, 0])]
        groups = data_tools.get_close_groups(x, 16)
        input_data = data_tools.sort_groups(groups)
        # input_data = data_tools.get_curvature_points(input_data, 64)

        input_data = np.array([input_data])

        tensor_data = torch.tensor(input_data).float().to(device)
        outputs = self.forward(tensor_data)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(outputs)):
            probabilities = torch.nn.functional.softmax(outputs[j], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for i in range(top5_prob.size(0)):
                print(class_names[top5_catid[i]], top5_prob[i].item())
        return predicted.cpu().numpy()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual = sys.path[0]
    dataset_dir = './64_16/train'
    val_dir = "./64_16/val"

    print("Loading dataset...")
    train_loader = ReadSet(dataset_dir, device)
    val_loader = ReadSet(val_dir, device)
    print("Dataset loaded")

    net = PEC(input_size=64, num_class=5, lr=1e-02)
    net.to(device)

    print("Starting training data....")

    train_loader = DataLoader(train_loader, batch_size=1024 * 256,
                              shuffle=False, num_workers=0)
    test_loader = DataLoader(val_loader, batch_size=1024 * 256,
                             shuffle=False, num_workers=0)

    print(net)
    print("Traning data loaded")
    print("Starting traning....")
    val_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="./",
        filename="PEC-{epoch:02d}-{val_loss:.2f}",
    )
    acc_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        dirpath="./",
        filename="PEC-{epoch:02d}-{val_acc:.2f}",
    )
    trainer = pl.Trainer(gpus=1, max_epochs=150, callbacks=[val_callback, acc_callback])
    #trainer = pl.Trainer(gpus=1, max_epochs=300, checkpoint_callback=False)

    trainer.fit(net, train_loader, test_loader)
    trainer.save_checkpoint("./modelPEC.ckpt")
    print("training finished")

    print("Best Acc: ", net.best_acc.detach().numpy())

    print("Done")

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Loss over time')
    ax1.plot(net.loss)
    ax2.set_title('Accuracy over time')
    ax2.plot(net.acc)
    plt.show()


if __name__ == '__main__':
    main()
