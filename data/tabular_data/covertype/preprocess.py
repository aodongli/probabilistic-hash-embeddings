import numpy as np

AREA_CODE = ['A1', 'A2', 'A3', 'A4']
SOIL_CODE = [f'S{i}' for i in range(1,40+1)]

def reverse_one_hot(one_hot, dic):
    one_hot = np.array(one_hot, dtype=np.int32)
    idx = np.where(one_hot == 1)[0]
    return dic[int(idx)]

fin = open('covtype.data', 'r')
new_data = []
for l in fin.readlines():
    new_l = []
    l = l.replace('\n', '')
    l = l.split(',')
    new_l += l[:10]

    # map area code
    ac = [reverse_one_hot(l[10:14], AREA_CODE)]
    new_l += ac

    # map soil code
    sc = [reverse_one_hot(l[14:54], SOIL_CODE)]
    new_l += sc

    # label
    new_l += [l[54]]

    new_data.append(','.join(new_l))

with open('proc_covtype.data', 'w') as fout:
    for l in new_data:
        fout.write(l)
        fout.write('\n')