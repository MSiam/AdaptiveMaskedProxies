import matplotlib.pyplot as plt
import sys
import numpy as np
import os

def parse_file(path):

    f = open(path, 'r')
    ious = []
    mious = []

    n_classes = None
    for line in f:
        if 'Task' in line:
            taski = int(line.split(' ')[1])
            n_classes = (taski+1)*2
        elif 'Mean' in line:
            continue
        else:
            ious.append(float(line.split(' ')[1]))
            if len(ious) == n_classes:
                mious.append(np.nanmean(np.array(ious)))
                ious = []
    return mious

cl_paths = sorted(os.listdir(sys.argv[1]))

mious_list = []
styles = ['-b', '-g', '-m', '-r', '-c', '-y']
legends = ['Naive #1', 'Naive #10', 'Imprint 0.05', 'Imprint 0.02', 'Imprint 0.01']
for i in range(5):
    mious = parse_file(sys.argv[1]+cl_paths[i])
    plt.plot(list(range(1,6)), mious, styles[i], marker='o')

plt.xticks(range(1,6))
plt.xlabel('Encountered Tasks')
plt.ylabel('mIoU')
plt.legend(legends)
plt.title('Continually Learning 2 New Classes in each New Task')
#plt.show()
plt.savefig('plot_cl.png')

