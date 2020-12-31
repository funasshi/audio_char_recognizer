import numpy as np
import matplotlib.pyplot as plt


def create_graph(loss_list, epochs):
    x = np.arange(epochs)
    plt.plot(x, loss_list, "r", label="loss")
    plt.savefig("loss.png")
