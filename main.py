import torch
from model import MyModel
from utils import LoadTrainData, LoadTestData, train, draw_graph, test, submit


def main(batch_size=0, epochs=0):
    model = MyModel()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_dataset = LoadTrainData(batch_size=batch_size)
    train_loader = train_dataset.train_load()
    valid_loader = train_dataset.valid_load()

    test_dataset = LoadTestData(batch_size=batch_size)
    test_loader = test_dataset.test_load()

    model, train_acc, train_loss, valid_acc, valid_loss = train(model=model, epochs=epochs, train_loader=train_loader,
                                                                valid_loader=valid_loader, optimizer=optimizer,
                                                                criterion=criterion,
                                                                lr_scheduler=lr_scheduler)

    draw_graph(train_acc, train_loss, valid_acc, valid_loss)
    predict = test(model, test_loader)

    submit(predict)


if __name__ == '__main__':
    batch_size = 128
    epochs = 0
    main(batch_size=batch_size, epochs=epochs)
