import torch
import torch.nn as nn


class Tokenizer():
    def __init__(self, tokenizer_type, token_len, stride, 
                 token_num, lag=None, padding_patch=None,
                 low_limit=-5.0, high_limit=5.0, 
                 #low_limit=-2, high_limit=2, 
                 n_bins=20, n_special_tokens=0, #4094
                ):
        # B = 4094 used in Chronos paper
        # bin centers c1 < . . . < cB on the real line where c1 = -15.0 and cB = 15.0

        self.tokenizer_type = tokenizer_type
        self.token_len = token_len
        self.stride = stride
        self.token_num = token_num
        self.lag = lag
        self.padding_patch = padding_patch
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.n_bins = n_bins
        self.n_special_tokens = n_special_tokens
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        self.centers = torch.linspace(
            low_limit,
            high_limit,
            n_bins - n_special_tokens - 1,
        )
        self.boundaries = torch.cat(
            (
                torch.tensor([-float("inf")]), 
                (self.centers[1:] + self.centers[:-1]) / 2, 
                torch.tensor([float("inf")])
            )
        )
        
    def _patch_fixed_len(self, z):
        """
        Creates a patches context from a time series tensor of shape [batch_size, nvars, seq_len].

        Parameters
        ----------
        series : torch.Tensor
            Input tensor of shape [batch_size, nvars, seq_len].
        token_len : int
            token length parameter. Specifies the step size for selecting elements.
        stride : int
            stride parameter. Specifies how many timestamps from the beginning of 
            the first patch to start the next patch.

        Returns
        -------
        torch.Tensor
            Patched tensor of shape [batch_size, nvars, patch_num, patch_len], 
            where selected_seq_len depends on `lag`.
        """

        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)

        patches = z.unfold(
                dimension=-1, size=self.token_len, step=self.stride
        ) #[bs x nvars x patch_num x patch_len]

        return patches

    def _lags(self, z):
        """
        Creates a lagged context from a time series tensor of shape [batch_size, nvars, seq_len].
        Selects elements from the end and works backward with a given lag.

        Parameters
        ----------
        series : torch.Tensor
            Input tensor of shape [batch_size, nvars, seq_len].
        lag : int
            Lag parameter. Specifies the step size for selecting elements.

        Returns
        -------
        torch.Tensor
            Lagged tensor of shape [batch_size, nvars, selected_seq_len], 
            where selected_seq_len depends on `lag`.
        """
        if self.lag <= 0:
            raise ValueError("Lag must be a positive integer.")

        # Reverse the sequence dimension (dim=-1)
        reversed_series = torch.flip(z, dims=[-1])
        # Select elements with the given lag
        lagged_series = reversed_series[..., ::self.lag]
        lags = torch.flip(lagged_series, dims=[-1])
        lags = lags.unsqueeze(-1)

        return lags
    
    def _bins(self, z):
        """
        Bins the data into uniform categories and returns categorical token IDs.
        Parameters:
        ----------
        z : torch.Tensor
            Input tensor of shape [batch_size, nvars, seq_len].
        Returns:
        -------
        torch.Tensor
            Token IDs of shape [batch_size, nvars, bins].
        """
        # Flatten last dimension for binning
        batch_size, nvars, seq_len = z.shape
        flattened_z = z.view(-1, seq_len)

        self.boundaries = self.boundaries.to(flattened_z.device)
        token_ids = torch.bucketize(
                flattened_z, self.boundaries, right=True)

        token_ids.clamp_(0, self.n_bins - 1)

        # Reshape back to original dimensions
        token_ids = token_ids.view(batch_size, nvars, seq_len)
        token_ids = token_ids.unsqueeze(-1) #[bs x nvars x seq_len x 1]
        token_ids = token_ids.float()

        return token_ids

    def unbin(self, token_ids):
        """
        Maps token IDs back to the corresponding bin centers.
        Parameters:
        ----------
        token_ids : torch.Tensor
            Tokenized data of shape [batch_size, nvars, seq_len].
        Returns:
        -------
        torch.Tensor
            Unbinned data of shape [batch_size, nvars, seq_len].
        """
         # Remove special tokens offset
        self.centers = self.centers.to(token_ids.device)
        indices = torch.clamp(
            token_ids - self.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )
        indices = indices.long()

        return self.centers[indices]
    
    def output_transform(self, z):
        if self.tokenizer_type == 'lags':
            output = self._lags(z)
            
        elif self.tokenizer_type == 'patch_fixed_length':
            output = self._patch_fixed_len(z)
            
        elif self.tokenizer_type == 'patch_adaptive_length':
            output = self._patch_fixed_len(z)
            
        elif self.tokenizer_type == 'bins': 
            output = self._bins(z)
            
        else:
            raise ValueError(f"Unsupported tokenizer_type: {self.tokenizer_type}")
            
        return output
