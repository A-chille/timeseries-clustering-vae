from vrae.vrae import VRAE
import time
import shutil
import numpy as np
import torch
import torch.nn as nn   
from torch.utils.data import DataLoader, TensorDataset

X_train = np.load(fr'X_train_long_size.npy', allow_pickle=True)
X_test = np.load(fr'X_test_long_size.npy', allow_pickle=True)
y_train = np.load(fr'y_train_long_size.npy', allow_pickle=True)
y_test = np.load(fr'y_test_long_size.npy', allow_pickle=True)

train_dataset = TensorDataset(torch.from_numpy(X_train).to('cuda'))
test_dataset = TensorDataset(torch.from_numpy(X_test))

print('X_train shape\t',X_train.shape,'\t X_test shape\t',X_test.shape)

for replicate in [0]:
    t0 = time.time()

    sequence_length = X_train.shape[1]

    number_of_features = 1
    dload = './model_dir_mm' #download directory
    hidden_size = 264
    hidden_layer_depth = 6
    batch_size = 200
    learning_rate = 0.0005 # 0.0005
    n_epochs = 400
    dropout_rate = 0.2
    optimizer = 'Adam' # options: ADAM, SGD
    cuda = True # options: True, False
    print_every=1000
    clip = True # options: True, False
    max_grad_norm=5
    loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
    block = 'LSTM' # options: LSTM, GRU


    for latent_length in range(50,50-1,-5):
        print('latent_length\t',latent_length,'\t elapsed time\t',(time.time()-t0)/60,' minutes\t',(time.time()-t0)/3600.0,' hours')

        vrae = VRAE(sequence_length=sequence_length,
                    number_of_features = number_of_features,
                    hidden_size = hidden_size, 
                    hidden_layer_depth = hidden_layer_depth,
                    latent_length = latent_length,
                    batch_size = batch_size,
                    learning_rate = learning_rate,
                    n_epochs = n_epochs,
                    dropout_rate = dropout_rate,
                    optimizer = optimizer, 
                    cuda = cuda,
                    print_every=print_every, 
                    clip=clip, 
                    max_grad_norm=max_grad_norm,
                    loss = loss,
                    block = block,
                    dload = dload)

        vrae.fit(train_dataset, test_dataset, save=True)

        file = open(fr'model_dir_mm/print.txt').readlines()
        train_losses = []
        test_losses = []
        for line in file:
            if 'Average loss' in line:
                train_losses.append(float(line.split(' ')[-2]))
                test_losses.append(float(line.split(' ')[-1][:-2]))

        shutil.move(fr'model_dir_mm/model.pth', fr'latent_long/model_{replicate}_{latent_length}.pth')
        shutil.move(fr'model_dir_mm/model_best.pth', fr'latent_long/model_best_{replicate}_{latent_length}.pth')
        np.save(fr'latent_long/train_losses_{replicate}_{latent_length}.npy', train_losses)
        np.save(fr'latent_long/test_losses_{replicate}_{latent_length}.npy', test_losses)