import numpy as np
import matplotlib.pyplot as plt


def create_loss_graph(train_loss_list, test_loss_list, epochs):
    fig = plt.figure()
    x = np.arange(epochs)
    plt.plot(x, train_loss_list, "r", label="train_loss")
    plt.plot(x, test_loss_list, "b", label="test_loss")
    plt.legend()
    plt.savefig("loss.png")


def create_acc_graph(train_acc_list, test_acc_list, topk_acc_list, epochs):
    fig = plt.figure()
    x = np.arange(epochs)
    plt.plot(x, train_acc_list, "r", label="train_acc")
    plt.plot(x, test_acc_list, "b", label="test_acc")
    plt.plot(x, topk_acc_list, "g", label="topk_acc")
    plt.legend()
    plt.savefig("acc.png")
