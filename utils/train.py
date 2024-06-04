from torch import nn
import torch
from utils.utils_set import *
from torch.utils import tensorboard as tb
def train(net, train_iter, test_iter, epochs, lr, device):
    """
    train a net in device;
    """
    def init_weight(net):
        if type(net) == nn.Conv2d or type(net) == nn.Linear:
            nn.init.xavier_normal_(net.weight)

    net.apply(init_weight)

    print("training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    writer = tb.SummaryWriter("./runs")
    for epoch in range(epochs):
        metric = Accumulator(3)
        net.train()
        for index,(X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
        train_l = metric[0]/metric[2]
        train_acc = metric[1]/metric[2]
        # if (index + 1) % (num_batches // 5) == 0 or index == num_batches - 1:
        print(f"epoch:{epoch}",f"train_loss:{train_l}", f"train_acc:{train_acc}")
        writer.add_scalar(tag="train_loss", scalar_value=train_l, global_step=epoch)
        writer.add_scalar(tag="train_accuracy", scalar_value=train_acc, global_step=epoch)
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        writer.add_scalar(tag="test_acc", scalar_value=test_acc, global_step=epoch)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
