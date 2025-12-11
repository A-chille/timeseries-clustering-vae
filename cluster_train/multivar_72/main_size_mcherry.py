from vrae.vrae import VRAE
import time
import shutil
import numpy as np
import torch
import torch.nn as nn   
from torch.utils.data import DataLoader, TensorDataset

X_train = np.load(fr'numpy_save_files/X_train_multivar_analysis_72.npy', allow_pickle=True)
X_test = np.load(fr'numpy_save_files/X_test_multivar_analysis_72.npy', allow_pickle=True)
y_train = np.load(fr'numpy_save_files/y_train_multivar_analysis_72.npy', allow_pickle=True)
y_test = np.load(fr'numpy_save_files/y_test_multivar_analysis_72.npy', allow_pickle=True)

print('X_train shape\t',X_train.shape,'\t X_test shape\t',X_test.shape)

# train_size = TensorDataset(torch.from_numpy(X_train[:,:,0,np.newaxis]).to('cuda'))
# test_size = TensorDataset(torch.from_numpy(X_test[:,:,0,np.newaxis]).to('cuda'))

# train_sos = TensorDataset(torch.from_numpy(X_train[:,:,1,np.newaxis]).to('cuda'))
# test_sos = TensorDataset(torch.from_numpy(X_test[:,:,1,np.newaxis]).to('cuda'))

# train_mcherry = TensorDataset(torch.from_numpy(X_train[:,:,2,np.newaxis]).to('cuda'))
# test_mcherry = TensorDataset(torch.from_numpy(X_test[:,:,2,np.newaxis]).to('cuda'))

# train_size_sos = TensorDataset(torch.from_numpy(X_train[:,:,:2]).to('cuda'))
# test_size_sos = TensorDataset(torch.from_numpy(X_test[:,:,:2]).to('cuda'))

train_size_mcherry = TensorDataset(torch.from_numpy(X_train[:,:,::2]).to('cuda'))
test_size_mcherry = TensorDataset(torch.from_numpy(X_test[:,:,::2]).to('cuda'))

# train_sos_mcherry = TensorDataset(torch.from_numpy(X_train[:, :,1:]).to('cuda'))
# test_sos_mcherry = TensorDataset(torch.from_numpy(X_test[:, :,1:]).to('cuda'))

# train_size_sos_mcherry = TensorDataset(torch.from_numpy(X_train).to('cuda'))
# test_size_sos_mcherry = TensorDataset(torch.from_numpy(X_test).to('cuda'))

train_dataset = train_size_mcherry
test_dataset = test_size_mcherry
folder = 'multivar_72_size_mcherry'
number_of_features = 2

for replicate in [0]:
    t0 = time.time()

    sequence_length = 72

    dload = folder #download directory
    hidden_size = 90
    hidden_layer_depth = 2
    batch_size = 100
    learning_rate = 0.0005 # 0.0005
    n_epochs = 200
    dropout_rate = 0.2
    optimizer = 'Adam' # options: ADAM, SGD
    cuda = True # options: True, False
    print_every=1000
    clip = True # options: True, False
    max_grad_norm=5
    loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
    block = 'LSTM' # options: LSTM, GRU


    for latent_length in range(14,5-1,-1):
        print('latent_length\t',latent_length,'\t ', 'number_of_features\t',number_of_features,'\t', 'hidden_size\t',hidden_size, '\t', 'hidden_layer_depth\t',hidden_layer_depth,'\t', 'batch_size\t',batch_size,'\t','learning_rate\t',learning_rate,'\t','n_epochs\t',n_epochs)
        print('elapsed time\t',(time.time()-t0)/60,' minutes\t',(time.time()-t0)/3600.0,' hours')

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

        file = open(fr'{folder}/print.txt').readlines()
        train_losses = []
        test_losses = []
        for line in file:
            if 'Average loss' in line:
                train_losses.append(float(line.split(' ')[-2]))
                test_losses.append(float(line.split(' ')[-1][:-2]))

        shutil.move(fr'{folder}/model.pth', fr'{folder}/model_{replicate}_{latent_length}.pth')
        shutil.move(fr'{folder}/model_best.pth', fr'{folder}/model_best_{replicate}_{latent_length}.pth')
        np.save(fr'{folder}/train_losses_{replicate}_{latent_length}.npy', train_losses)
        np.save(fr'{folder}/test_losses_{replicate}_{latent_length}.npy', test_losses)
        print('elapsed time\t',(time.time()-t0)/60,' minutes\t',(time.time()-t0)/3600.0,' hours')