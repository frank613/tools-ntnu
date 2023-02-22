import numpy as np
import pdb


auc_mono_cmu = np.array([0.826, 0.872, 0.671, 0.91, 0.72, 0.892, 0.832, 0.899, 0.899, 0.82, 0.889, 0.783, 0.912, \
                     0.914, 0.744, 0.641])

auc_mono_libri = np.array([0.933, 0.957, 0.805, 0.949, 0.989, 0.825, 0.979, 0.97, 0.906, 0.91, 0.97, 0.981, 0.985, \
                     0.816, 0.988, 0.985])

auc_mono_ted = np.array([0.898, 0.903, 0.733, 0.735, 0.91, 0.843, 0.84, 0.927, 0.937, 0.883, 0.919, 0.923, 0.948, \
                     0.83, 0.72, 0.916])


print(np.mean(auc_mono_libri), np.mean(auc_mono_ted), np.mean(auc_mono_cmu))
print(np.std(auc_mono_libri), np.std(auc_mono_ted), np.std(auc_mono_cmu))


def compute_entropy(in_list, minV=-20, maxV=0, nBin=50):

    copied = in_list.copy()
    for n,value in enumerate(copied):
        if value < minV:
            copied[n] = minV
        elif value > maxV:
            copied[n] = maxV
        else:
            pass
    hist1 = np.histogram(copied, bins=nBin, range=(minV,maxV), density=True)
    stats = hist1[0]
    stats = stats[stats!=0]
    stats = stats/stats.sum()
    ent = -(stats*np.log(np.abs(stats))).sum()
    return (ent,len(stats))

print(compute_entropy(auc_mono_libri, 0, 0.9 ))

