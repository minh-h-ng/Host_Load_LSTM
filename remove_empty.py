import csv
import os
import shutil

data_path = '/home/minh/Desktop/host_LSTM/datasets/full/'
out_path = '/home/minh/Desktop/host_LSTM/datasets/full_fixed/'

filelist = os.listdir(data_path)
filelist.sort()

out_count = 0
for filename in filelist:
    with open(data_path + filename,'r') as f:
        reader = csv.reader(f)
        zero_count = 0
        for line in reader:
            if float(line[0]) == 0:
                zero_count += 1
        if zero_count>=1000:
            print('file has too many zeros: ', filename)
        else:
            shutil.copyfile(data_path + filename, out_path + filename)
            out_count += 1
print('total number of files retained: ', out_count)