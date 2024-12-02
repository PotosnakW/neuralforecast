import torch
import torch.nn as nn


class Tokenizer():
    def __init__(self, tokenizer_type, token_len, stride, 
                 token_num, lag=None, padding_patch=None):

        self.tokenizer_type = tokenizer_type
        self.token_len = token_len
        self.stride = stride
        self.token_num = token_num
        self.lag = lag
        self.padding_patch = padding_patch
        
        if padding_patch=="end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        
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
    
    def output_transform(self, z):
        if self.tokenizer_type == 'lags':
            output = self._lags(z)
            
        elif self.tokenizer_type == 'patch_fixed_length':
            if self.padding_patch == "end":
                z = self.padding_patch_layer(z)
            output = self._patch_fixed_len(z)
            
        return output

    
    
    
    
# class MeanScaleUniformBinsTokenizer():
#     def __init__(
#         self, low_limit: float, high_limit: float, config: ChronosConfig
#     ) -> None:
#         self.config = config
#         self.centers = torch.linspace(
#             low_limit,
#             high_limit,
#             config.n_tokens - config.n_special_tokens - 1,
#         )
#         self.boundaries = torch.concat(
#             (
#                 torch.tensor([-1e20], device=self.centers.device),
#                 (self.centers[1:] + self.centers[:-1]) / 2,
#                 torch.tensor([1e20], device=self.centers.device),
#             )
#         )

#     def _input_transform(
#         self, context: torch.Tensor, scale: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         attention_mask = ~torch.isnan(context)

#         if scale is None:
#             scale = torch.nansum(
#                 torch.abs(context) * attention_mask, dim=-1
#             ) / torch.nansum(attention_mask, dim=-1)
#             scale[~(scale > 0)] = 1.0

#         scaled_context = context / scale.unsqueeze(dim=-1)
#         token_ids = (
#             torch.bucketize(
#                 input=scaled_context,
#                 boundaries=self.boundaries,
#                 # buckets are open to the right, see:
#                 # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
#                 right=True,
#             )
#             + self.config.n_special_tokens
#         )

#         token_ids.clamp_(0, self.config.n_tokens - 1)

#         token_ids[~attention_mask] = self.config.pad_token_id

#         return token_ids, attention_mask, scale

#     def _append_eos_token(
#         self, token_ids: torch.Tensor, attention_mask: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size = token_ids.shape[0]
#         eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id)
#         token_ids = torch.concat((token_ids, eos_tokens), dim=1)
#         eos_mask = torch.full((batch_size, 1), fill_value=True)
#         attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

#         return token_ids, attention_mask

#     def context_input_transform(
#         self, context: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         length = context.shape[-1]

#         if length > self.config.context_length:
#             context = context[..., -self.config.context_length :]

#         token_ids, attention_mask, scale = self._input_transform(context=context)

#         if self.config.use_eos_token and self.config.model_type == "seq2seq":
#             token_ids, attention_mask = self._append_eos_token(
#                 token_ids=token_ids, attention_mask=attention_mask
#             )

#         return token_ids, attention_mask, scale

#     def label_input_transform(
#         self, label: torch.Tensor, scale: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         length = label.shape[-1]

#         assert length == self.config.prediction_length
#         token_ids, attention_mask, _ = self._input_transform(context=label, scale=scale)

#         if self.config.use_eos_token:
#             token_ids, attention_mask = self._append_eos_token(
#                 token_ids=token_ids, attention_mask=attention_mask
#             )

#         return token_ids, attention_mask

#     def output_transform(
#         self, samples: torch.Tensor, scale: torch.Tensor
#     ) -> torch.Tensor:
#         scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
#         indices = torch.clamp(
#             samples - self.config.n_special_tokens - 1,
#             min=0,
#             max=len(self.centers) - 1,
#         )
#         return self.centers[indices] * scale_unsqueezed

