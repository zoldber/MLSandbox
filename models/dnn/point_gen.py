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

ax = plt.axes(xlim=(0, 16), ylim=(0, 16))

# Plot boundary functions
ax.plot(X, F)
ax.plot(X, G)

# Fill regions between
ax.fill_between(X, ([0] * 17), F)
ax.fill_between(X, G, ([16] * 17))
ax.fill_between(X, F, G)

ax.set_yticks([0, 4, 8, 12, 16])
ax.set_ylabel("Input Feature A")

ax.set_xticks([0, 4, 8, 12, 16])
ax.set_xlabel("Input Feature B")

ax.set_title("Classification Boundaries for Inputs [A, B]")

# Generate random points and label by position
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


# show plot before writing to csv
plt.show()

with open('test_data/samples.csv', 'w') as f:
    for sample in samples:
        f.write(f'{sample[0]:.3f}, {sample[1]:.3f}\n')

with open('test_data/labels.csv', 'w') as f:
    for label in labels:
        f.write(f'{label[0]}, {label[1]}, {label[2]}\n')
