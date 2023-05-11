import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import StepLR
from segment import BatchSegment

class GraphVAEv4:
    """Final undirected graphvae version for anomaly detection, add graph learning loss
    fixed memory overflow bugs

    Args:
        window_size (int). Sliding window size.
        channels (int): Channel count of the input signals.
        name (str, optional): Model name. Defaults to 'VAE_Dense'.
        num_epochs (int, optional): Epochs for model training. Defaults to 256.
        batch_size (int, optional): Batch size for model training. Defaults to 256.
        lr ([type], optional): Learning rate. Defaults to 1e-3.
        lr_decay (float, optional): Learning rate decay. Defaults to 0.8.
        clip_norm_value (float, optional): Gradient clip value. Defaults to 12.0.
        weight_decay ([type], optional): L2 regularization. Defaults to 1e-3.
        h_dim (int, optional): Hidden dim between x and z for VAE's encoder and decoder Defaults to 200.
        z_dim (int, optional): Defaults to 20.

    Attributes:
        model: VAE model built by torch.
    """

    def __init__(self,
                 window_size,
                 channels,
                 name='GraphVAEv4',
                 num_epochs=256,
                 batch_size=64,
                 lr=1e-3,
                 lr_decay=0.8,
                 clip_norm_value=12.0,
                 weight_decay=1e-3,
                 h_dim=200,
                 z_dim=20,
                 device=torch.device('cpu'),
                 lambda_gcn=0.5,
                 neighbour=10,
                 graph_alpha=3.):
        super(GraphVAEv4, self).__init__()
        self._name = name
        self._window_size = window_size
        self._channels = channels
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._lr_decay = lr_decay
        self._clip_norm_value = clip_norm_value
        self._weight_decay = weight_decay        
        self._h_dim = h_dim
        self._z_dim = z_dim
        self._device = device
        self._lambda_gcn = lambda_gcn
        self._neighbour = neighbour
        self._graph_alpha = graph_alpha

        self._vae = GraphVAE_Module(self._window_size, self._channels,
                              self._h_dim,
                              self._z_dim, self._lambda_gcn, self._device, k=self._neighbour, alpha=self._graph_alpha).to(self._device)
        self._trained = False
        
    def fit(self, X):
        """Train the model

        Args:
            X (array_like): The input 2-D array.
                            The first dimension denotes timesteps.
                            The second dimension denotes the signal channels.
        """
        def loss_vae(x_hat, x, mu, logvar, adjmat):
            # print(x_hat.shape)
            mse_x = F.mse_loss(x_hat, x)
            mse_x = mse_x * x.shape[1] * x.shape[2]
            # KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            KLD = 1 + logvar - mu ** 2 - logvar.exp()
        #     print(KLD.shape)
            KLD = torch.sum(KLD, dim=-1)
        #     print(KLD.shape)
            KLD = torch.sum(KLD, dim=-1)
        #     print(KLD.shape)
            KLD *= -0.5
        #     print(KLD.shape)
        #     print((mse_x + KLD).shape)
            mse_reg = F.mse_loss(x, torch.matmul(adjmat, x)) * x.shape[1] * x.shape[2]
            # L1_reg = torch.sum(torch.abs(adjmat))
            
            return mse_x, torch.mean(KLD), mse_reg # , L1_reg

        train_gen = BatchSegment(len(X),
                                 self._window_size,
                                 self._batch_size,
                                 shuffle=True,
                                 discard_last_batch=True)

        self._vae.train()
        optimizer = torch.optim.Adam(self._vae.parameters(),
                                     lr=self._lr,
                                     weight_decay=self._weight_decay)
        scheduler = StepLR(optimizer, step_size=10,
                           gamma=self._lr_decay)

        for epoch in range(self._num_epochs):
            # i = 0
            for (batch_x, ) in train_gen.get_iterator([np.asarray(X, dtype=np.float32)]):
                optimizer.zero_grad()
                
                batch_x = torch.from_numpy(batch_x).to(self._device).permute(0, 2, 1)

                x_hat, mu, logvar, x_mean, x_logvar, adjmat = self._vae(batch_x)
                mse_x, kld, mse_reg= loss_vae(x_hat, batch_x, mu, logvar, adjmat)
                
                total_loss = mse_x + kld
                total_loss += mse_reg
                # total_loss += L1_reg

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._vae.parameters(),
                                              self._clip_norm_value)
                optimizer.step()

                # if i % 90 == 0:
                #     print("Epoch[{}/{}] loss: {:.8f} MSE_x: {:.8f}, KLD: {:.8f}, MSE_reg: {:.8f}".format(epoch+1,
                #                                                                         self._num_epochs,
                #                                                                         total_loss.item(),
                #                                                                         mse_x.item(),
                #                                                                         kld.item(),
                #                                                                         mse_reg.item(),
                #                                                                         ))
                # i += 1
    
            scheduler.step()

        self._trained = True

    def detect(self, X):
        """ Get anomaly score of input sequence.

        Args:
            X (array_like): Input sequence.

        Returns:
            A dict with attributes:
                origin_series: ndarray [timesteps, channels], Origin time series
                recon_series: ndarray [timesteps, channels], Reconstruct time series
                score: ndarray [timesteps, channels], Corresponding anomaly score.
        """

        def data_to_windows(data, window_size, window_step=1):
            windows = np.array([data[i: i + window_size] for i in range(0, len(data) - window_size + 1, window_step)], dtype=np.float32)
            return windows
        
        def get_expectation(model, mean, logvar, sample_count):
            z_list = list()
            for _ in range(sample_count):
                z = model.reparameterize(mean, logvar)
                z_list.append(z.cpu().detach().numpy())
            z_list = torch.from_numpy(np.array(z_list, dtype=np.float32))
        
            z = torch.mean(z_list, dim=0).to(self._device)

            return z

        def get_recon_x(model, test_data, n_z=200):
        
            h_x = model.encoder(test_data)
            h_z = model.h_z_net(h_x)

            adj = model.get_adjmat()
            merge_h_z = self._lambda_gcn * torch.matmul(adj, h_z) + (1 - self._lambda_gcn) * h_z
            
            z_mean = model.z_mean_net(merge_h_z)
            # z_mean [batch, channels, z_dim]
            
            z_logvar = model.z_logvar_net(merge_h_z)
            z = get_expectation(model, z_mean, z_logvar, n_z)
            # z [batch, channels, z_dim]
            h_zdec = model.decoder(z)
            x_mean = model.x_mean_net(h_zdec)
            x_logvar = model.x_logvar_net(h_zdec)

            return x_mean

        self._vae.eval()
        # test_windows = data_to_windows(X, self._window_size)
        # test_windows_input = torch.from_numpy(test_windows).to(self._device).permute(0, 2, 1)
        # x_mean = get_recon_x(self._vae, test_windows_input)

        test_gen = BatchSegment(len(X),
                                 self._window_size,
                                 self._batch_size,
                                 shuffle=False,
                                 discard_last_batch=False)
        x_mean = list()
        for (batch_x, ) in test_gen.get_iterator([np.asarray(X, dtype=np.float32)]):
            batch_x = torch.from_numpy(batch_x).to(self._device).permute(0, 2, 1)
            batch_x_mean = get_recon_x(self._vae, batch_x)
            x_mean.append(batch_x_mean.cpu().detach().numpy())

        x_mean = np.concatenate(x_mean, axis=0)
        # print(x_mean.shape)
        origin_signals = X[self._window_size - 1:, :]
        # print('in v2')
        # print(origin_signals.shape)
        recon_signals = x_mean[:, :, -1]
        # print(recon_signals.shape)
        ano_score = np.square(origin_signals - recon_signals)

        return {'origin': origin_signals,
                'recon': recon_signals,
                'score': ano_score}

    def predict(self, X):
        pass

    def forecast(self, X):
        pass

    def save(self, path):
        torch.save(self._vae.state_dict(), path)

    def load(self, path):
        self._vae.load_state_dict(torch.load(path))

    def to(self, device):
        self._device = device
        self._vae = self._vae.to(self._device)

    def get_adjmat(self):
        self._vae.eval()
        return self._vae.get_adjmat().cpu().detach().numpy()

def to_var(x, device):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return Variable(x.to(device))

class GraphVAE_Module(nn.Module):
    def __init__(self, window_size, channels, h_dim, z_dim, lambda_gcn, device, node_dim=30, k=10, alpha=3):
        super(GraphVAE_Module, self).__init__()

        self.channels = channels
        self.k = k
        self._lambda_gcn = lambda_gcn
        self._device = device
        self._alpha = alpha

        self.encoder = nn.Sequential(
            nn.Linear(window_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
#             nn.Linear(h_dim, z_dim * 2)
        )
        
        self.h_z_net = nn.Sequential(
            nn.Linear(h_dim, z_dim),
        )

        self.z_mean_net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
        )
        
        self.z_logvar_net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            # nn.Softplus(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        
        self.x_mean_net = nn.Sequential(
            nn.Linear(h_dim, window_size),
        )
        
        self.x_logvar_net = nn.Sequential(
            nn.Linear(h_dim, window_size),
            # nn.Softplus(),
        )

        # undirected
        self.node_emb1 = nn.Embedding(channels, node_dim)
        # self.node_emb2 = nn.Embedding(channels, node_dim)
        self.emb_lin1 = nn.Linear(node_dim, node_dim)
        # self.emb_lin2 = nn.Linear(node_dim, node_dim)

        self.idx = torch.arange(channels).to(self._device)

        # print('v3 core params: lambda-{}, k-{}, alpha-{}'.format(self._lambda_gcn, self.k, self._alpha))
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()), self._device)
        z = mu + std * esp
        return z
    
    def get_adjmat(self):
        eb1 = self.node_emb1(self.idx)
        eb2 = self.node_emb1(self.idx)

        eb1 = torch.tanh(self._alpha * self.emb_lin1(eb1))
        eb2 = torch.tanh(self._alpha * self.emb_lin1(eb2))

        a = torch.mm(eb1, eb2.T)
        adj = F.relu(self._alpha * torch.tanh(a))
        mask = torch.zeros(self.channels, self.channels).to(self._device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        adj = adj + torch.eye(adj.size(0)).to(self._device)
        d = adj.sum(1)
        adj = adj / d.view(-1, 1)
        return adj

    def forward(self, x):
        '''
        x:[batch, channels, window_size]
        '''        

        h_x = self.encoder(x)
        h_z = self.h_z_net(h_x)
        
        adj = self.get_adjmat()
        merge_h_z = self._lambda_gcn * torch.matmul(adj, h_z) + (1 - self._lambda_gcn) * h_z

        z_mean = self.z_mean_net(merge_h_z)
        # z_mean [batch, channels, z_dim]
        
        z_logvar = self.z_logvar_net(merge_h_z)
        z = self.reparameterize(z_mean, z_logvar)
        # z [batch, channels, z_dim]
        h_zdec = self.decoder(z)
        x_mean = self.x_mean_net(h_zdec)
        x_logvar = self.x_logvar_net(h_zdec)
        x_hat = self.reparameterize(x_mean, x_logvar)

        return x_hat, z_mean, z_logvar, x_mean, x_logvar, adj
