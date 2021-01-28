import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from math import cos, pi

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Z Axes')


def f(x, y):
    return 10 * 2 + ((x ** 2 - 10 * cos(2 * pi * x)) + (y ** 2 - 10 * cos(2 * pi * y)))


def plot(population, colour):
    x = [individual.gene[0] for individual in population]
    y = [individual.gene[1] for individual in population]
    z = [individual.fitness for individual in population]

    ax.set_zticks(np.arange(0, 20, step=0.5))

    ax.scatter3D(x, y, z, s=200, color=colour)


def plot_best(population, colour):
    for individual in population:
        ax.scatter3D(individual.gene[0], individual.gene[1], individual.fitness, s=400, marker="x", color='g')


def plot_optimum():
    ax.scatter3D([0], [0], [1], color='b', s=800, marker='p')


def show_plot():
    fig.savefig('fig1.png', dpi=300)
    # plt.show()

