import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import operator

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


runs = ['1385', '1386', '1387', '1388', '1389']

mious_runs = []
for i, run in enumerate(runs):
    cl_paths = sorted(os.listdir(sys.argv[1]+str(i)+'/'))
    mious_list = []
    for j in range(len(cl_paths)):
        mious = parse_file(sys.argv[1]+str(i)+'/'+cl_paths[j])
        mious_list.append(mious)
    mious_runs.append(mious_list)

styles = ['-b', '-r']
legends = ['Naive #1', 'Imprint 0.01']

mious_list = np.array(mious_runs).mean(0)
stds_list = np.array(mious_runs).std(0)
clrs = ['blue', 'red']
for i, (mious, stds) in enumerate(zip(mious_list, stds_list)):
    if i == 0:
        c = 0
        plt.plot(list(range(1,6)), mious, styles[c], marker='o')
        plt.fill_between(range(1,6), mious-stds, mious+stds, facecolor=clrs[c], alpha=0.2)
    elif i == 4:
        c = 1
        plt.plot(list(range(1,6)), mious, styles[c], marker='o')
        plt.fill_between(range(1,6), mious-stds, mious+stds, facecolor=clrs[c], alpha=0.2)

axes = plt.subplot(1, 1, 1)
plt.xlabel('Encountered Tasks')
plt.ylabel('mIoU')

plt.xticks(range(1,6))
plt.xlabel('Encountered Tasks')
plt.ylabel('mIoU')
plt.legend(legends)
plt.title('iPASCAL: Continually Learning 2 New Classes in each Task')

#plt.show()
plt.savefig('ipascal_std.png')

