import numpy as np
import matplotlib.pyplot as plt


def create_loss_graph(loss_list, epochs):
    fig = plt.figure()
    x = np.arange(epochs)
    plt.plot(x, loss_list, "r", label="loss")
    plt.savefig("loss.png")


def create_acc_graph(train_acc_list, test_acc_list, epochs):
    fig = plt.figure()
    x = np.arange(epochs)
    plt.plot(x, train_acc_list, label="train_acc")
    plt.plot(x, test_acc_list, label="test_acc")
    plt.savefig("acc.png")
