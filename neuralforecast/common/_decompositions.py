import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalDecomposition:
    """
    Applies hierarchical decomposition for trend-seasonality or NHITS interpolation.
    """

    def __init__(self, mode: str, **kwargs):
        super(HierarchicalDecomposition, self).__init__()
        assert mode in [None, 'dlinear_trend_seasonality', 'fourier_bases', 'nhits_interpolation'], \
            "Mode must be 'dlinear_trend_seasonality' or 'nhits_interpolation'."

        self.mode = mode
        if mode == 'dlinear_trend_seasonality':
            self.decomp_block = SeriesDecomp(kernel_size=kwargs.get('moving_avg_window', 3))
        elif mode == 'fourier_bases':
            self.decomp_block = FourierDecomp(top_k=kwargs.get('top_k', 3))
        elif mode == 'nhits_interpolation':
            self.decomp_block = IdentityBasis(
                backcast_size=kwargs['backcast_size'],
                forecast_size=kwargs['forecast_size'],
                interpolation_mode=kwargs['interpolation_mode'],
                out_features=kwargs.get('out_features', 1)
            )

    def output_transform(self, x):
         
        if self.mode == 'dlinear_trend_seasonality':
            backcast, residual = self.decomp_block(x)
            backcasts = [backcast, residual]
        elif self.mode == 'fourier_bases':
            backcast, residual = self.decomp_block(x)
            backcasts = backcast+[residual]
        
        backcasts = torch.stack(backcasts, dim=1)  # [bs, num_decomps, nvars, seq_len]
        backcasts = backcasts.view(-1, x.shape[1], x.shape[2])  # [bs * num_decomps, 1, seq_len]
        backcasts = backcasts.real

        return backcasts


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride, padding=0):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.padding = 0
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=self.padding)

    def forward(self, x):
        # padding on the both ends of time series
        context_len = x.shape[-1]
        pad = (self.kernel_size - 1) // 2
        output_len = ((context_len+2*pad) + 2*self.padding - self.kernel_size) // self.stride + 1
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)

        diff = context_len - output_len
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2 + diff)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.MovingAvg = MovingAvg(kernel_size, stride=1)
            
    def forward(self, x):
        moving_mean = self.MovingAvg(x)
        residual = x - moving_mean
        return moving_mean, residual
    
    
class FourierDecomp(nn.Module):
    def __init__(
        self,
        top_k=10,
    ):
        super().__init__()
        self.top_k = top_k
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        ts = 1.0/seq_len
        t = torch.arange(1e-5,1,ts)

        dft = torch.fft.fft(input=x.squeeze(1), n=seq_len, dim=-1, norm='ortho') 
        ks = torch.arange(seq_len) 

        dft_magnitudes = torch.abs(dft)  # Compute the absolute value
        sorted_indices = torch.argsort(dft_magnitudes, descending=True) 
        top_kvals = sorted_indices[:, :self.top_k]
        top_dft = torch.gather(dft, dim=1, index=top_kvals)

        top_kvals = top_kvals.unsqueeze(-1).repeat(1, 1, seq_len)
        top_dft = top_dft.unsqueeze(-1).repeat(1, 1, seq_len)
        
        top_kvals = top_kvals.to(x.device)
        t = t.to(x.device)

        bs = torch.exp(-2j * torch.pi * top_kvals * t)
        bs = torch.flip(bs, dims=(2,))
        basis_functions = (top_dft  * bs)/torch.sqrt(torch.tensor(seq_len))
        #basis_functions = bs/torch.sqrt(torch.tensor(seq_len)) # [bs, num_decomps, seq_len]

        #correct for shift of 1
        basis_functions = F.pad(basis_functions, (1, 0), mode='replicate', value=0)
        basis_functions = basis_functions[:, :, :seq_len]

        basis_functions = basis_functions.unsqueeze(dim=-2) # [bs, num_decomps, 1, seq_len]
        residual = x - torch.sum(basis_functions, dim=1) # [bs, 1, seq_len]
        
        basis_functions = list(torch.unbind(basis_functions, dim=1))        

        return basis_functions, residual
