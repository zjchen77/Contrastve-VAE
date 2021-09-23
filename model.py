import torch
import torch.nn as nn
import torch.nn.functional as F
from .types_ import *
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self,
                 in_channels: int,
                 
                 
                 hidden_dims: list = None,
                 feature_dim: int =128,
                 **kwargs) -> None:
        super(Model, self).__init__()
        

        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        in_channels = 3 # To account for the extra label channel
        #Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        nn.Linear(hidden_dims[-1]*4, feature_dim)
        
        self.encoder = nn.Sequential(*modules)
        
        
        

        #Decoder
        modules = []



        self.decoder_input = nn.Linear(feature_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh()())

                
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    
    
    def forward(self, x):
        mu = self.encoder(x)
        logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        
        out = torch.flatten(z, start_dim=1)   #原feature

        recon_x = self.decoder(z)
        recon_x = self.final_layer(recon_x)
        return F.normalize(out, dim=-1),  recon_x, mu, logvar
