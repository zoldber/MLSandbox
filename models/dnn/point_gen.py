from random import random
import matplotlib.pyplot as plt


# boundary function
def f(x):

    val = 0.02*(x**3) - 0.15*(x**2) - x + 8

    return min(max(val, 0), 15)


bx = [x for x in range(16)]
by = list(map(f, bx))

# plot boundary function
plt.plot(bx, by)

# generate random points and label by position
# relative to the boundary defined above
list_size = 100

samples = []
labels = []

for _ in range(list_size):
    x = random() * 15.0
    y = random() * 15.0

    if y > f(x):
        label = (1.0, 0.0)
    else:
        label = (0.0, 1.0)

    samples.append((x, y))
    labels.append(label)

    plt.scatter(x, y, color='green' if label[0] else 'blue')

# show plot before writing to csv
plt.show()

with open('samples.csv', 'w') as f:
    for sample in samples:
        f.write(f'{sample[0]}, {sample[1]}\n')

with open('labels.csv', 'w') as f:
    for label in labels:
        f.write(f'{label[0]}, {label[1]}\n')
