import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

'''
Boundary planes should be stored in their own folder, and
should have a common prefix ("bp_0" -> "bp_n" for n fies in path)
'''

path = "../../boundary_planes/"

boundaryFilePrefix = "bpc_"

numFiles = 0

for file in os.listdir(path):
    if file.startswith(boundaryFilePrefix):
        numFiles += 1

# Height and width of plane in cells (not pixels)
planeH = (0, 16)
planeW = (0, 16)

colors = ["black", "gray", "white"]

fig = plt.figure()

ax = plt.axes(xlim=planeW, ylim=planeH)

ax.set_yticks(np.arange(0, 17, 4))
ax.set_ylabel("Input Feature A")

ax.set_xticks(np.arange(0, 17, 4))
ax.set_xlabel("Input Feature B")

ax.set_title("Network Classification of Input [A, B]")

def animate(i):

    x = 0
    y = 0

    with open(path + boundaryFilePrefix + str(i) + ".csv") as f:

        y = 0

        for line in f:

            line = line.strip("\n")
            line = line.split(", ")

            x = 0

            for color in line:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color))
                x += 1

            y += 1

    return []

boundFitAnim = FuncAnimation(fig, animate, save_count=0, frames=numFiles, interval=10, blit=True)

boundFitAnim.save("fit.gif", writer="imagemagick")
