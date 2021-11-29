import os
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score

from utils import RMSELoss
from data import ToyDataset
from model import Encoder, Decoder


## Hyperparameter
intermidiate_dim = 32
batch_size = 64
latent_dim = 8
epochs = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
mse = nn.MSELoss()
rmse = RMSELoss()
is_cuda = True
k = 10 # k-fold
Tensor = torch.FloatTensor
save_folder = 'results'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

## Prepare data
training_feature = np.loadtxt('dataset/X.txt')
Y = np.loadtxt('dataset/Y.txt')
ground_truth_r = Y

np.random.seed(seed=0)

# np.random.seed(0)
skf = StratifiedKFold(n_splits=k)
pred = np.zeros((ground_truth_r.shape))
fake = np.zeros((ground_truth_r.shape[0]))
fake[:300] = 1  


# 2) model
encoder = Encoder(intermidiate_dim=32, latent_dim=8)
decoder = Decoder(intermidiate_dim=32, latent_dim=8)
if is_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# 3) optimizer
optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), 
                             lr=lr, betas=(b1, b2))
## Start training
iterations = 1
for train_idx, test_idx in skf.split(training_feature, fake):
    print("========== {0}th fold ==========".format(iterations))
    # split tr/tt dataset
    training_feature_sk = training_feature[train_idx,:]
    training_score = ground_truth_r[train_idx].reshape(-1, 1)

    testing_feature_sk = training_feature[test_idx,:]
    testing_score = ground_truth_r[test_idx].reshape(-1, 1)

    # 1) data
    tr_dataset = ToyDataset(training_feature_sk, training_score)
    tt_dataset = ToyDataset(testing_feature_sk, testing_score)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True) 
    tt_loader = DataLoader(tt_dataset, batch_size=batch_size, shuffle=False) 
    
    # 4) train
    for epoch in range(epochs):
        # train phase
        encoder.train()
        decoder.train()
        total_vae_loss = 0
        for i, data in enumerate(tr_loader):
            x, target = data
            if is_cuda:
                x = x.cuda()
                target = target.cuda()
            # forward propagation
            [mu_z, logvar_z, z], [mu_y, logvar_y, y], z_bar_y = encoder(x)
            x_hat = decoder(z)

            # compute loss & backpropagation 
            reconstruction_loss = mse(x, x_hat)
            kl_loss = 1 + logvar_z - torch.square(mu_z - z_bar_y) - torch.exp(logvar_z)
            kl_loss = -0.5 * torch.sum(kl_loss)
            label_loss = torch.divide(0.5 * torch.square(mu_y - target), torch.exp(logvar_y)) + 0.5 *logvar_y
            vae_loss = torch.mean(reconstruction_loss + kl_loss + label_loss)
            
            # vanila regression
            # vae_loss = mse(mu_y, target)
            
            total_vae_loss += vae_loss.item()
            vae_loss.backward()
            optimizer.step()

        # test phase
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            x = Tensor(testing_feature_sk).cuda()
            [mu_z, logvar_z, z], [mu_y, logvar_y, y], z_bar_y = encoder(x)
            pred[test_idx] = np.array(mu_y.cpu().detach())[:,0]
            rmse_loss = rmse(mu_y, Tensor(testing_score).cuda())
        # print
        print("[Epoch: %d/%d] [Train loss: %.3f] ---> [Test RMSE: %.3f]" \
              % (epoch + 1, epochs, total_vae_loss/(i + 1), rmse_loss.item()))
    iterations += 1


## Validation
print("Mean squared error: %.3f" % mean_squared_error(ground_truth_r, np.array(pred)))
print('R2 Variance score: %.3f' % r2_score(ground_truth_r, np.array(pred)))

# Plot Prediction vs. Ground-truth Y
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(ground_truth_r, np.array(pred),  color='black')
plt.xlabel('ground truth')
plt.ylabel('prediction truth')
ax.axis('equal')
plt.savefig('results/true_vs_pred.png')
plt.close()

# Visualize latent space
encoder.eval()
decoder.eval()
with torch.no_grad():
    [mu_z, logvar_z, z], [mu_y, logvar_y, y], z_bar_y = encoder(Tensor(training_feature).cuda())

tsne = MDS(n_components=2, random_state=0)
X_2d = tsne.fit_transform(np.array(mu_z.cpu().detach()))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_2d[:, 0], X_2d[:, 1], c=ground_truth_r)
plt.title('TSNE visualization of latent space')
ax.axis('equal')
plt.savefig('results/tsne.png')