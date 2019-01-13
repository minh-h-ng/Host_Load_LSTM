import csv
import numpy as np

#result_file_1 = './results/val500_group2_random_seq24_hidden128_layer1_batch39_stateless'
result_file_1 = './results/LSTM_best'
#result_file_2 = './results/val500_group1_random_seq24_hidden128_layer1_batch39_stateless'
result_file_2 = './results/LSTM_real'

results1 = []
with open(result_file_1, 'r') as f:
    for line in f:
        if line[:3]=='Ave':
            #results1.append(float(line[4:].strip()))
            results1.append(float(line.split(':')[-1]))

print('len:',len(results1))
print('average:',np.average(results1))

results2 = []
with open(result_file_2, 'r') as f:
    for line in f:
        if line[:3]=='Ave':
            #results1.append(float(line[4:].strip()))
            results2.append(float(line.split(':')[-1]))
#results2 = results2[::-1]

results2 = results2[:len(results1)]

print('len:',len(results2))
print('average:',np.average(results2))
