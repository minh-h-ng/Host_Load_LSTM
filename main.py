import logging
import argparse
import json
import glob
import os
import utils
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil
import sys

from tqdm import tqdm as tqdm
from models import VanillaLSTM
from datasets import HostDataset
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

# Initialize logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("lstmconfig", help="path to LSTM config file", type=str)
parser.add_argument("outbase", help="path to output base folder", type=str)
parser.add_argument("predictiondir", help="path to prediction folder", type=str)
args = parser.parse_args()

lstmconfig = json.load(open(args.lstmconfig + '.json', 'r'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataset_loader, seq_len, criterion, optimizer):
    model = model.train()

    # tracking losses
    losses = utils.AverageMeter()

    for i, (x, y) in enumerate(dataset_loader):
        # (batch, seq_len, input_size)
        x = x.contiguous().view(-1, seq_len, 1).to(device)
        y = y.contiguous().view(-1, lstmconfig['output_length']).to(device)

        # forward-pass
        prediction = model(x)

        # calculate loss
        loss = criterion(prediction, y)

        # update losses
        losses.update(loss.item(), torch.numel(x))

        # reset gradients
        #optimizer.zero_grad()

        # backprop and update weights
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

    # return avg loss for this epoch
    return losses.avg

def validate(model, dataset_loader, seq_len, criterion, scaler):
    model = model.eval()

    # tracking losses
    losses = utils.AverageMeter()

    real = np.empty((0, lstmconfig['output_length']), np.float64)
    pred = np.empty((0, lstmconfig['output_length']), np.float64)

    with torch.no_grad():
        for i, (x, y) in enumerate(dataset_loader):
            # (batch, seq_len, input_size)
            x = x.contiguous().view(-1, seq_len, 1).to(device)
            y = y.contiguous().view(-1, lstmconfig['output_length']).to(device)

            # forward-pass
            prediction = model(x)

            real = np.vstack((real, scaler.inverse_transform(y.cpu())))
            pred = np.vstack((pred, scaler.inverse_transform(prediction.cpu())))

            loss = mean_squared_error(real, pred)

            losses.update(loss.item(), torch.numel(y))

    # return avg loss
    return losses.avg, real, pred

def forecast(model, x):
    model = model.eval()

    with torch.no_grad():
        # forward-pass
        prediction = model(x)
        return prediction

def main():
    logger.info('PROGRAM START!')
    logger.info('lstmconfig: ' + str(lstmconfig))
    logger.info('outbase:' + str(args.outbase))
    logger.info('predictiondir: ' + str(args.predictiondir))

    datadir = args.outbase + '/datasets/' + lstmconfig['mode'] + str(lstmconfig['number_of_hosts']) + '_' + 'group' \
              + str(lstmconfig['hosts_per_group']) + '_' + lstmconfig['cluster']
    hosts = glob.glob(os.path.join(datadir, '*.csv'))
    hosts = sorted(hosts, key = lambda k : int(k.split('/')[-1].split('.')[0]))

    # over-all metrics
    mses = utils.AverageMeter()

    predictionDir = args.predictiondir + '/' + args.lstmconfig.split('/')[-1]
    if os.path.exists(predictionDir) and os.path.isdir(predictionDir):
        shutil.rmtree(predictionDir)
    os.makedirs(predictionDir)

    # process all hosts:
    for host in hosts:
        csv_name = os.path.basename(host)
        logger.info('Processing ' + format(csv_name))

        # network
        model = VanillaLSTM(lstmconfig).to(device)
        """model = VanillaLSTM(lstmconfig)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)"""

        n_test_days = 3

        # training dataset
        train_dataset = HostDataset(host, lstmconfig['sequence_length'], 'train', n_test_days,
                                    lstmconfig['output_length'], lstmconfig['hosts_per_group'], preprocessing.StandardScaler())
        train_loader = data.DataLoader(train_dataset, lstmconfig['batch_size'], shuffle=True)

        # test dataset
        test_dataset = HostDataset(host, lstmconfig['sequence_length'], 'test', n_test_days,
                                   lstmconfig['output_length'], lstmconfig['hosts_per_group'], preprocessing.StandardScaler())
        test_loader = data.DataLoader(test_dataset, lstmconfig['batch_size'])

        learning_rate = 0.05
        # criteria
        criterion = nn.MSELoss()
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # epoch-0
        best_train_loss = train(model, train_loader, lstmconfig['sequence_length'], criterion, optimizer)
        best_val_loss, best_real, best_pred = \
            validate(model, test_loader, lstmconfig['sequence_length'], criterion, train_dataset.get_scaler())

        counter = 0

        epoch_count = 1
        # epoch-1,2,3...
        with tqdm(range(1, lstmconfig['n_max_epochs']), desc="Epoch") as iterator:
            for epoch in iterator:
                epoch_count += 1
                if epoch_count % 30 == 0:
                    for g in optimizer.param_groups:
                        g['lr'] /= 10
                # train
                train_loss = train(model, train_loader, lstmconfig['sequence_length'], criterion, optimizer)

                # validate
                val_loss, cur_real, cur_pred = \
                    validate(model, test_loader, lstmconfig['sequence_length'], criterion, train_dataset.get_scaler())

                if lstmconfig['eval'] == 'best':
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_real = cur_real
                        best_pred = cur_pred
                elif lstmconfig['eval'] == 'real':
                    best_val_loss = val_loss
                    best_real = cur_real
                    best_pred = cur_pred
                else:
                    logger.info('eval config not recognizable!')
                    sys.exit(0)

        predictionFile = predictionDir + '/' + host.split('/')[-1]
        results = np.concatenate((best_real, best_pred), axis=1)
        df = pd.DataFrame(results)
        df.to_csv(predictionFile, index=False, header=False)

        print("Average MSE: {}".format(best_val_loss))

    # Average RMSE
    #print("Average MSE: {}".format(mses.avg))


if __name__ == '__main__':
    main()