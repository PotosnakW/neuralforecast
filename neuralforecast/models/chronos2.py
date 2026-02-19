import logging
from typing import Optional

import torch
from chronos import BaseChronosPipeline

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE

logger = logging.getLogger(__name__)


class Chronos2(BaseModel):
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
        univariate: bool = True,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 0.0,
        device_map: str = "cuda",
        cpus: Optional[int] = None,
        alias: Optional[str] = None,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 0,
        learning_rate: float = 1e-4,
        num_lr_decays: int = 0,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 1,
        batch_size: int = 64,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 64,
        inference_windows_batch_size: Optional[int] = 1_024,
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
        super(Chronos2, self).__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            loss=loss,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=0,
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

        self.univariate = univariate
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.device_map = device_map
        self.cpus = cpus

        logger.info(f"Initializing Chronos2 model with alias: {alias}")
        logger.info(f"Horizon: {h}, n_series: {n_series}, univariate: {univariate}")

        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=self.device_map,
            top_k=self.top_k,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self.quantiles = self.pipeline.model.quantiles.tolist()
        

    def _median_forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        input_device = x_enc.device
        x_enc_cpu = x_enc.detach().to(device="cpu", dtype=torch.float32).contiguous()

        forecast = self.pipeline.predict(x_enc_cpu, prediction_length=self.h)
        forecast = torch.stack(forecast, dim=0)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2

        return forecast[:, :, median_idx, :].to(input_device)

    def forward(self, windows_batch, **kwargs):
        x = windows_batch["insample_y"]  # [B, L, N]
        B, L, N = x.shape
        
        if self.univariate:
            x_uni = x.permute(2, 0, 1).reshape(N * B, 1, L) #[N*B, 1, L]
            y_uni = self._median_forecast(x_uni).squeeze(1) # [N*B, H]
            forecast = y_uni.reshape(N, B, self.h).permute(1, 2, 0) # [B, H, N]
            return forecast

        x_enc = x.permute(0, 2, 1).contiguous()  # [B, N, L]
        forecast = self._median_forecast(x_enc)  # [B, N, H]
        forecast = forecast.permute(0, 2, 1).contiguous() # [B, H, N]
        return forecast
