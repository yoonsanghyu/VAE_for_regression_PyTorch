import torch
import torch.nn as nn

## Define model
class Encoder(nn.Module):
    def __init__(self, intermidiate_dim, latent_dim):
        super(Encoder, self).__init__()
        # Build encoder model
        self.drop = nn.Dropout(0.25)
        self.dense1 = nn.Linear(301, 128)
        self.act1 = nn.Tanh()
        self.dense2 = nn.Linear(128, intermidiate_dim)
        self.act2 = nn.Tanh()

        self.model = nn.Sequential(self.dense1, self.act1,
                                   self.dense2, self.act2)
        
        # Posterior on Y; probabilistic regressor
        self.dense_mu_y = nn.Linear(intermidiate_dim, 1)
        self.dense_logvar_y = nn.Linear(intermidiate_dim, 1)

        # q(z|x)
        self.dense_mu_z = nn.Linear(intermidiate_dim, latent_dim)
        self.dense_logvar_z = nn.Linear(intermidiate_dim, latent_dim)

        # latent generator
        self.dense_gen_z = nn.Linear(1, latent_dim)

    def sampling(self, mu, log_var):
        # Reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        out = self.model(x)
        # z
        mu_z = self.dense_mu_z(out)
        logvar_z = self.dense_logvar_z(out)
        z = self.sampling(mu_z, logvar_z)
        # y
        mu_y = self.dense_mu_y(out)
        logvar_y = self.dense_logvar_y(out)
        y = self.sampling(mu_y, logvar_y)

        # y conditional z
        z_bar_y = self.dense_gen_z(y)

        return [mu_z, logvar_z, z], [mu_y, logvar_y, y], z_bar_y

class Decoder(nn.Module):
    def __init__(self, intermidiate_dim, latent_dim):
        super(Decoder, self).__init__()

        self.dense1 = nn.Linear(latent_dim, intermidiate_dim)
        self.act1 = nn.Tanh()
        self.dense2 = nn.Linear(intermidiate_dim, 128)
        self.act2 = nn.Tanh()
        self.dense3 = nn.Linear(128, 301)

        self.model = nn.Sequential(self.dense1, self.act1,
                                   self.dense2, self.act2,
                                   self.dense3)
    def forward(self, x):
        return self.model(x)