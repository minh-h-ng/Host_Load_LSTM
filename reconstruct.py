import csv
import json
import random
import os
import shutil
import sys
import numpy as np
import pandas as pd

# Store data outside of project folder due to long Pycharm index time
basedir = '/home/minh/Desktop/host_LSTM/'

configFile = './configs/reconstruct.json'
data_path = 'datasets/host_full/'

config = json.load(open(configFile,'r'))

def main():
    number_of_hosts = config['number_of_hosts']
    hosts_per_group = config['hosts_per_group']
    cluster = config['cluster']

    random.seed(0)
    filelist = os.listdir(basedir + data_path)
    filelist = sorted(filelist, key=lambda k: int(k.split('.')[0].split('_')[-1]))

    random_hosts = random.sample(filelist, number_of_hosts)
    out_path = 'datasets/val' + str(number_of_hosts) + '_' + 'group' + str(hosts_per_group) + '_' + str(cluster) + '/'
    if os.path.exists(basedir + out_path) and os.path.isdir(basedir + out_path):
        shutil.rmtree(basedir + out_path)
    os.makedirs(basedir + out_path)
    if hosts_per_group == 1:
        for host in random_hosts:
            curname = host.split('_')[-1]
            shutil.copyfile(basedir + data_path + host, basedir + out_path + curname)
    else:
        make_group(random_hosts, out_path, config)

def make_group(filelist, out_path, config):
    hosts_per_group = config['hosts_per_group']
    if not(len(filelist) % hosts_per_group == 0):
        print('Number of files not divisible by hosts_per_group!')
        sys.exit(0)
    no_of_group = int(len(filelist) / hosts_per_group)
    random.shuffle(filelist)
    print('path:',basedir + out_path)
    for i in range(no_of_group):
        curlist = filelist[i::no_of_group]
        outname = str(curlist[0].split('.')[0])
        for j in range(1,len(curlist)):
            outname += '_' + str(curlist[j].split('.')[0])
        datas = None
        for j in range(len(curlist)):
            cur_data = pd.read_csv(basedir + data_path + curlist[j])
            if datas is None:
                datas = cur_data
            else:
                datas = pd.concat([datas,cur_data],axis=1,sort=False)
        datas.to_csv(basedir + out_path + outname + '.csv', index=False)


        #print('list:',filelist[i::no_of_group])


if __name__ == '__main__':
    main()