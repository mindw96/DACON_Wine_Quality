import copy
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, dataset=None, labels=None, state=True):
        self.state = state
        if self.state:
            self.dataset = dataset.drop(['id', 'quality'], axis=1)
            self.labels = labels
        else:
            self.dataset = dataset.drop(['id'], axis=1)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx].values
        data = torch.Tensor(data)
        if self.state:
            label = self.labels.iloc[idx]
            label = torch.Tensor([int(label)])
            return data, label
        else:
            return data

    def __len__(self):
        return len(self.dataset)


def train_preprocessing(train):
    word_to_num = {"white": 0, "red": 1}
    train['type'] = train['type'].replace(word_to_num)

    num_to_num = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}
    train['quality'] = train['quality'].replace(num_to_num)

    train['fixed acidity'] = (train['fixed acidity'] - train['fixed acidity'].min(axis=0)) / (
            train['fixed acidity'].max(axis=0) - train['fixed acidity'].min(axis=0))

    train['residual sugar'] = (train['residual sugar'] - train['residual sugar'].min(axis=0)) / (
            train['residual sugar'].max(axis=0) - train['residual sugar'].min(axis=0))

    train['free sulfur dioxide'] = (train['free sulfur dioxide'] - train['free sulfur dioxide'].min(axis=0)) / (
            train['free sulfur dioxide'].max(axis=0) - train['free sulfur dioxide'].min(axis=0))

    train['total sulfur dioxide'] = (train['total sulfur dioxide'] - train['total sulfur dioxide'].min(axis=0)) / (
            train['total sulfur dioxide'].max(axis=0) - train['total sulfur dioxide'].min(axis=0))

    train['pH'] = (train['pH'] - train['pH'].min(axis=0)) / (train['pH'].max(axis=0) - train['pH'].min(axis=0))

    train['alcohol'] = (train['alcohol'] - train['alcohol'].min(axis=0)) / (
            train['alcohol'].max(axis=0) - train['alcohol'].min(axis=0))

    return train


def test_preprocessing(test):
    word_to_num = {"white": 0, "red": 1}
    test['type'] = test['type'].replace(word_to_num)

    test['fixed acidity'] = (test['fixed acidity'] - test['fixed acidity'].min(axis=0)) / (
            test['fixed acidity'].max(axis=0) - test['fixed acidity'].min(axis=0))

    test['residual sugar'] = (test['residual sugar'] - test['residual sugar'].min(axis=0)) / (
            test['residual sugar'].max(axis=0) - test['residual sugar'].min(axis=0))

    test['free sulfur dioxide'] = (test['free sulfur dioxide'] - test['free sulfur dioxide'].min(axis=0)) / (
            test['free sulfur dioxide'].max(axis=0) - test['free sulfur dioxide'].min(axis=0))

    test['total sulfur dioxide'] = (test['total sulfur dioxide'] - test['total sulfur dioxide'].min(axis=0)) / (
            test['total sulfur dioxide'].max(axis=0) - test['total sulfur dioxide'].min(axis=0))

    test['pH'] = (test['pH'] - test['pH'].min(axis=0)) / (test['pH'].max(axis=0) - test['pH'].min(axis=0))

    test['alcohol'] = (test['alcohol'] - test['alcohol'].min(axis=0)) / (
            test['alcohol'].max(axis=0) - test['alcohol'].min(axis=0))

    return test


class LoadTrainData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train = pd.read_csv('dataset/train.csv')
        self.train = train_preprocessing(self.train)

        self.trainset = self.train.sample(frac=0.8)
        self.validset = self.train.sample(frac=0.2)
        self.train_label = self.trainset['quality']
        self.valid_label = self.validset['quality']

    def train_load(self):
        train_dataset = CustomDataset(self.trainset, self.train_label, state=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return train_loader

    def valid_load(self):
        valid_dataset = CustomDataset(self.validset, self.valid_label, state=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return valid_loader


class LoadTestData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.test = pd.read_csv('dataset/test.csv')
        self.test = test_preprocessing(self.test)

    def test_load(self):
        test_dataset = CustomDataset(self.test, state=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return test_loader


def ACCURACY(true, pred):
    score = np.mean(true == pred)
    return score


def train(model=None, epochs=1, train_loader=None, valid_loader=None, optimizer=None, criterion=None,
          lr_scheduler=None):
    best_acc = 0.0
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []

    device = torch.device("cuda:0")
    model = model.to(device)

    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        model.train()

        train_batch_loss, valid_batch_loss, train_batch_acc, valid_batch_acc = 0.0, 0.0, 0.0, 0.0
        batch_index = 0

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            label = torch.max(label, 1)[0].long()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(data)

                loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            acc = ACCURACY(label.detach().cpu().numpy(), output.detach().cpu().numpy().argmax(-1))

            train_batch_loss += loss.item()
            train_batch_acc += acc

            batch_index += 1

        train_loss.append(train_batch_loss / batch_index)
        train_acc.append(train_batch_acc / batch_index)

        lr_scheduler.step()

        # Validation
        model.eval()

        batch_index = 0

        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            label = torch.max(label, 1)[0].long()
            optimizer.zero_grad()

            with torch.no_grad():
                output = model(data)

                loss = criterion(output, label)

            acc = ACCURACY(label.detach().cpu().numpy(), output.detach().cpu().numpy().argmax(-1))

            valid_batch_loss += loss.item()
            valid_batch_acc += acc

            batch_index += 1

        valid_loss.append(valid_batch_loss / batch_index)
        valid_acc.append(valid_batch_acc / batch_index)

        # 1 Epoch Result
        print('Train Acc: {:.2f} Valid Acc: {:.2f}'.format(train_acc[epoch] * 100, valid_acc[epoch] * 100))
        print('Train Loss: {:.4f} Valid Loss: {:.4f}'.format(train_loss[epoch], valid_loss[epoch]))

        if valid_acc[epoch] > best_acc:
            best_idx = epoch
            best_acc = valid_acc[epoch]
            torch.save(model.state_dict(), 'check.pt')
            base_cnn_best_model_wts = copy.deepcopy(model.state_dict())
            print('==> best model saved - {} / {:.4f}'.format(best_idx + 1, best_acc))

    print()
    print('Best valid Acc: %d - %.4f' % (best_idx + 1, best_acc))

    # load best model weights
    model.load_state_dict(base_cnn_best_model_wts)
    torch.save(model.state_dict(), 'Final.pt')
    print('final model saved')

    return model, train_acc, train_loss, valid_acc, valid_loss


def draw_graph(train_acc, train_loss, valid_acc, valid_loss):
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-')
    ax1.plot(valid_acc, 'r-')
    ax1.set_xlabel('epoch')

    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-')
    ax2.plot(valid_loss, 'k-')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    plt.show()


def test(model=None, test_loader=None):
    model.eval()

    predicts = []

    for data in test_loader:
        output = model(data)
        predicts.append(output.data.numpy())

    predict = []
    for tensor in predicts:
        for row in tensor:
            pred = np.argmax(row)
            if pred == 0:
                predict.append(4)
            elif pred == 1:
                predict.append(5)
            elif pred == 2:
                predict.append(6)
            elif pred == 3:
                predict.append(7)
            elif pred == 4:
                predict.append(8)

    return predict


def submit(predict):
    submission = pd.read_csv('dataset/sample_submission.csv')
    submission['quality'] = predict
    submission.to_csv("submission.csv", index=False)
