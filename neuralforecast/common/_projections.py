import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """Multi-Layer Perceptron with Skip Connection

    **Parameters:**<br>
    `in_features`: int, dimension of input.<br>
    `out_features`: int, dimension of output.<br>
    `activation`: str, activation function to use.<br>
    `hidden_size`: int, dimension of hidden layers.<br>
    `num_layers`: int, number of hidden layers.<br>
    `dropout`: float, dropout rate.<br>
    """

    def __init__(
        self, individual, n_vars, nf, h, c_out=1, head_dropout=0, activation="GELU",
    ):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.nf = nf
        self.h = h
        self.c_out = c_out
        self.activation = getattr(nn, activation)()
        self.dropout = head_dropout
        self.use_skip = False
        
    def linear_head(self):
        layers = [
                nn.Flatten(start_dim=-2),
                nn.Linear(self.nf, self.h * self.c_out),
                nn.Dropout(self.dropout)
            ]
        
        self.layers = nn.Sequential(*layers)
        
    def residual_head(self, num_layers=1):
        layers = [
                nn.Flatten(start_dim=-2),
                nn.Linear(self.nf, self.nf),
                self.activation,
                nn.Dropout(self.dropout)
            ]
        for i in range(num_layers):
            layers += [
                nn.Linear(in_features=self.nf, out_features=self.nf),
                self.activation,
                nn.Dropout(self.dropout),
            ]
            # Output layer
        layers += [nn.Linear(in_features=self.nf, out_features=self.h * self.c_out)]

        self.use_skip == True
        self.layers = nn.Sequential(*layers)
        
    def projection_layer(self, head_type):
        if head_type == "linear":
            self.linear_head()
        elif head_type == "residual":
            self.residual_head()
            
        return self

    def forward(self, x):
        residual = x.clone()

        x = self.layers(x)
        if self.use_skip:
            x += residual  # Add skip connection

        return x
    

class ProjectionEmbd(nn.Module):
    """Multi-Layer Perceptron with Skip Connection

    **Parameters:**<br>
    `in_features`: int, dimension of input.<br>
    `out_features`: int, dimension of output.<br>
    `activation`: str, activation function to use.<br>
    `hidden_size`: int, dimension of hidden layers.<br>
    `num_layers`: int, number of hidden layers.<br>
    `dropout`: float, dropout rate.<br>
    """

    def __init__(
        self, individual, n_vars, nf, h, c_out=1, dropout=0, activation="GELU",
    ):
        super().__init__()

        activation="ReLU"
        
        self.individual = individual
        self.n_vars = n_vars
        self.nf = nf
        self.h = h
        self.c_out = c_out
        self.activation = getattr(nn, activation)()
        self.dropout = dropout
        self.use_skip = False
        
    def linear_embd(self):
        self.W_P = nn.Linear(
            self.nf, self.h
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        
    def residual_embd(self, num_layers=1):
        layers = [
                nn.Linear(self.nf, self.nf),
                self.activation,
                nn.Dropout(self.dropout)
            ]
        for i in range(num_layers):
            layers += [
                nn.Linear(in_features=self.nf, out_features=self.nf),
                self.activation,
                nn.Dropout(self.dropout),
            ]
            # Output layer
        layers += [nn.Linear(in_features=self.nf, out_features=self.h * self.c_out)]

        self.use_skip == True
        self.W_P = nn.Sequential(*layers)
        
    def projection_layer(self, head_type):
        if head_type == "linear":
            self.linear_embd()
        elif head_type == "residual":
            self.residual_embd()

        return self

    def forward(self, x):
        residual = x.clone()

        x = self.W_P(x)
        if self.use_skip:
            x += residual  # Add skip connection

        return x


#class Flatten_Head(nn.Module):
# class ProjectionHead(nn.Module):
#     """
#     Flatten_Head
#     """

#     def __init__(self, individual, n_vars, nf, h, c_out, head_dropout=0):
#         super().__init__()

#         self.individual = individual
#         self.n_vars = n_vars
#         self.c_out = c_out

#         if self.individual:
#             self.linears = nn.ModuleList()
#             self.dropouts = nn.ModuleList()
#             self.flattens = nn.ModuleList()
#             for i in range(self.n_vars):
#                 self.flattens.append(nn.Flatten(start_dim=-2))
#                 self.linears.append(nn.Linear(nf, h * c_out))
#                 self.dropouts.append(nn.Dropout(head_dropout))
#         else:
#             self.flatten = nn.Flatten(start_dim=-2)
#             self.linear = nn.Linear(nf, h * c_out)
#             self.dropout = nn.Dropout(head_dropout)

#     def forward(self, x):  # x: [bs x nvars x hidden_size x patch_num]
#         if self.individual:
#             x_out = []
#             for i in range(self.n_vars):
#                 z = self.flattens[i](x[:, i, :, :])  # z: [bs x hidden_size * patch_num]
#                 z = self.linears[i](z)  # z: [bs x h]
#                 z = self.dropouts[i](z)
#                 x_out.append(z)
#             x = torch.stack(x_out, dim=1)  # x: [bs x nvars x h]
#         else:
#             x = self.flatten(x)
#             x = self.linear(x)
#             x = self.dropout(x)
#         return x