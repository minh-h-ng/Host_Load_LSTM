import os
import shutil

data_path = '/home/minh/Desktop/host_full_backup/'
out_path = '/home/minh/Desktop/host_LSTM/datasets/host_full/'

filelist = os.listdir(data_path)
filelist = sorted(filelist, key = lambda k : int(k.split('.')[0].split('_')[-1]))

for filename in filelist:
    outname = filename.split('_')[-1]
    shutil.copyfile(data_path + filename, out_path + outname)
