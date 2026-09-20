"""Zero-shot wrappers for foundation models that live outside the MICA conda env.

Toto-2 and TimesFM-3 ship dependencies that conflict with envs/neuralforecast
(the env currently serving the gate sweep), so each runs from its own env. These
classes deliberately live in mica/training/ rather than in neuralforecast/models/:
adding them to that package's __init__ would make `import neuralforecast` require
toto2/timesfm everywhere, breaking every MICA training job.

Both subclass BaseModel and return [B, H, N] from forward(), exactly like
neuralforecast/models/chronos2.py, so NeuralForecast.cross_validation drives them
through the identical windowing as every other baseline -- the forecasts.csv that
comes out is directly comparable, with no windowing logic reimplemented here.

The third-party import happens inside __init__, never at module import time, so
this file can be imported from any env for introspection.
"""
import logging
from typing import Optional

import torch

from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE

logger = logging.getLogger(__name__)

# Both models emit the same 9 quantile levels.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class _ZeroShotBase(BaseModel):
    """Shared plumbing: a frozen, max_steps=0 multivariate BaseModel."""

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        n_series: int,
        checkpoint: str,
        device_map: str = "cuda",
        batch_size: int = 64,
        windows_batch_size: int = 64,
        inference_windows_batch_size: Optional[int] = 1024,
        alias: Optional[str] = None,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 0,
        learning_rate: float = 1e-4,
        num_lr_decays: int = 0,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 1,
        valid_batch_size: Optional[int] = None,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            loss=loss,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=0,  # zero-shot: never trained
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )
        self.checkpoint = checkpoint
        self.device_map = device_map
        self.median_idx = QUANTILE_LEVELS.index(0.5)


class Toto2(_ZeroShotBase):
    """Datadog Toto-2, zero-shot.

    forecast() wants (batch, n_variates, time_steps) and returns
    (9, batch, n_variates, horizon) over QUANTILE_LEVELS; we take the median.
    """

    def __init__(self, *args, checkpoint: str = "Datadog/Toto-2.0-22m",
                 decode_block_size: int = 768, **kwargs):
        super().__init__(*args, checkpoint=checkpoint, **kwargs)
        self.decode_block_size = decode_block_size

        from toto2 import Toto2Model  # noqa: PLC0415 -- see module docstring

        logger.info(f"Loading Toto2 checkpoint {checkpoint}")
        device = torch.device(self.device_map if torch.cuda.is_available() else "cpu")
        self.toto = Toto2Model.from_pretrained(checkpoint).to(device).eval()
        # forecast() reduces the context with "... (seq patch) -> ... seq", so the
        # context length must be a whole number of patches.
        self.patch_size = int(self.toto.config.patch_size)

    @torch.no_grad()
    def forward(self, windows_batch, **kwargs):
        x = windows_batch["insample_y"]  # [B, L, N]
        target = x.permute(0, 2, 1).contiguous()  # [B, N, L]
        device = target.device
        mask = torch.ones_like(target, dtype=torch.bool)

        # MICA sets input_size = h * multiplier, which for short horizons is not a
        # multiple of patch_size (ett1/W: 16 vs 32). Left-pad up to a whole patch
        # and mark the pad unobserved rather than shrinking/growing input_size --
        # every other model in the sweep sees the same context, so changing it here
        # would silently break comparability. forecast() maps an all-unobserved
        # patch to series id -1, i.e. ignores it.
        rem = target.shape[-1] % self.patch_size
        if rem:
            pad = self.patch_size - rem
            target = torch.nn.functional.pad(target, (pad, 0))
            mask = torch.nn.functional.pad(mask, (pad, 0), value=False)

        quantiles = self.toto.forecast(
            {
                "target": target,
                "target_mask": mask,
                "series_ids": torch.zeros(
                    target.shape[0], target.shape[1], dtype=torch.long, device=device
                ),
            },
            horizon=self.h,
            decode_block_size=self.decode_block_size,
            has_missing_values=bool(rem),
        )
        median = quantiles[self.median_idx]  # [B, N, H]
        return median.permute(0, 2, 1).contiguous().to(x.device)  # [B, H, N]


class TimesFM3(_ZeroShotBase):
    """Google TimesFM-3, zero-shot, multivariate.

    TimesFM-3 carries variate attention (use_variate_attention=True by default),
    so the whole [B, N, L] window goes through decode() in one pass and the N
    channels attend across each other -- the same multivariate treatment Toto2
    gets. The high-level TimesFM3Forecaster.predict_batch() takes a list of 1-D
    contexts and would forecast each channel in isolation, throwing that away, so
    it is deliberately not used here.

    decode() returns (B, N, H, n_quantiles) -- note the quantile axis is LAST,
    where Toto2 puts it FIRST.
    """

    def __init__(self, *args, checkpoint: str = "google/timesfm-3.0-pytorch", **kwargs):
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

        import timesfm  # noqa: PLC0415 -- see module docstring

        logger.info(f"Loading TimesFM3 checkpoint {checkpoint}")
        device = torch.device(self.device_map if torch.cuda.is_available() else "cpu")
        self.tfm = timesfm.TimesFM3Torch.from_pretrained(checkpoint).to(device).eval()
        # Its own quantile list, rather than assuming QUANTILE_LEVELS.
        self.median_idx = list(self.tfm.quantiles).index(0.5)

    @torch.no_grad()
    def forward(self, windows_batch, **kwargs):
        x = windows_batch["insample_y"]  # [B, L, N]
        target = x.permute(0, 2, 1).contiguous()  # [B, N, L]

        # No manual padding: decode() left-pads the context to input_patch_len
        # itself. No target_mask either -- it defaults to all-False, and for
        # TimesFM3 False means OBSERVED (it pads the mask with True to mark
        # padding). That is the opposite polarity to Toto2, where True means
        # observed; passing ones here would mask out the entire context.
        out = self.tfm.decode(target=target, horizon=self.h)
        if isinstance(out, tuple):
            out = out[0]

        median = out[..., self.median_idx]  # [B, N, H]
        return median.permute(0, 2, 1).contiguous().to(x.device)  # [B, H, N]
