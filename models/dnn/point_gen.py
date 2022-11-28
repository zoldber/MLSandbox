from random import random
import matplotlib.pyplot as plt


# boundary function
def f(x):

    val = 0.02*(x**3) - 0.35*(x**2) + x + 8.0

    return min(max(val, 0), 16)


def g(x):

    val = 8.0 + ((6.0 - (0.75 * x))**2)

    return min(max(val, 0), 16)

X = [xi for xi in range(17)]
F = list(map(f, X))
G = list(map(g, X))

# plot boundary functions
plt.plot(X, F)
plt.plot(X, G)


# generate random points and label by position
# relative to the boundary defined above
listSize = 16000

samples = []
labels = []

for _ in range(listSize):
    x = random() * 16.0
    y = random() * 16.0

    if y > g(x):
        label = [1.0, 0.0, 0.0]
    elif y > f(x):
        label = [0.0, 1.0, 0.0]
    else:
        label = [0.0, 0.0, 1.0]

    samples.append([x, y])
    labels.append(label)

if listSize <= 100:

    colors = ['orange', 'green', 'blue']

    numPoints = len(samples)

    for i in range(numPoints):

        plt.scatter(
                    x=samples[i][0], 
                    y=samples[i][1],
                    color=colors[labels[i].index(1.0)]
                    )


# show plot before writing to csv
plt.show()

with open('test_data/samples.csv', 'w') as f:
    for sample in samples:
        f.write(f'{sample[0]}, {sample[1]}\n')

with open('test_data/labels.csv', 'w') as f:
    for label in labels:
        f.write(f'{label[0]}, {label[1]}, {label[2]}\n')
