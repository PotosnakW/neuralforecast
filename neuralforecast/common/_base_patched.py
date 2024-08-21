# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/common.base_windows.ipynb.

# %% auto 0
__all__ = ['BasePatch']

# %% ../../nbs/common.base_windows.ipynb 5
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from ._base_model import BaseModel
from ._scalers import TemporalNorm
from ..tsdataset import TimeSeriesDataModule
from ..utils import get_indexer_raise_missing

# %% ../../nbs/common.base_windows.ipynb 6
class BasePatchED(BaseModel):
    """Base Windows

    Base class for all windows-based models. The forecasts are produced separately
    for each window, which are randomly sampled during training.

    This class implements the basic functionality for all windows-based models, including:
    - PyTorch Lightning's methods training_step, validation_step, predict_step.<br>
    - fit and predict methods used by NeuralForecast.core class.<br>
    - sampling and wrangling methods to generate windows.
    """

    def __init__(
        self,
        h,
        patch_len,
        stride,
        input_size,
        loss,
        valid_loss,
        learning_rate,
        max_steps,
        val_check_steps,
        batch_size,
        valid_batch_size,
        windows_batch_size,
        inference_windows_batch_size,
        start_padding_enabled,
        step_size=1,
        num_lr_decays=0,
        early_stop_patience_steps=-1,
        scaler_type="identity",
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        num_workers_loader=0,
        drop_last_loader=False,
        random_seed=1,
        alias=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            random_seed=random_seed,
            loss=loss,
            valid_loss=valid_loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            max_steps=max_steps,
            early_stop_patience_steps=early_stop_patience_steps,
            **trainer_kwargs,
        )

        # Padder to complete train windows,
        # example y=[1,2,3,4,5] h=3 -> last y_output = [5,0,0]
        self.h = h

        self.patch_len = torch.minimum(torch.tensor(patch_len), 
                                       torch.tensor(input_size + stride)
                                      )
        ## WILLA'S FIX SO H IS PREDICTED IF PATCH_LEN > HORIZON. 
        ## REMOVE torch.tensor(h) TO PREDICT PATCH_LEN ALWAYS
        self.patch_len = torch.minimum(self.patch_len, 
                                       torch.tensor(h)
                                      )
        
        self.input_size = input_size
        self.windows_batch_size = windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        if start_padding_enabled:
            self.padder_train = nn.ConstantPad1d(
                padding=(self.input_size - 1, self.h), value=0
            )
        else:
            #self.padder_train = nn.ConstantPad1d(padding=(0, self.h), value=0)
            self.padder_train = nn.ConstantPad1d(padding=(0, self.patch_len), value=0)

        # Batch sizes
        self.batch_size = batch_size
        if valid_batch_size is None:
            self.valid_batch_size = batch_size
        else:
            self.valid_batch_size = valid_batch_size
        if inference_windows_batch_size is None:
            self.inference_windows_batch_size = windows_batch_size
        else:
            self.inference_windows_batch_size = inference_windows_batch_size

        # Optimization
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.num_lr_decays = num_lr_decays
        self.lr_decay_steps = (
            max(max_steps // self.num_lr_decays, 1) if self.num_lr_decays > 0 else 10e7
        )
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.windows_batch_size = windows_batch_size
        self.step_size = step_size

        self.exclude_insample_y = exclude_insample_y

        # Scaler
        self.scaler = TemporalNorm(
            scaler_type=scaler_type,
            dim=1,  # Time dimension is 1.
            num_features=1 + len(self.hist_exog_list) + len(self.futr_exog_list),
        )

        # Fit arguments
        self.val_size = 0
        self.test_size = 0

        # Model state
        self.decompose_forecast = False

        # DataModule arguments
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        # used by on_validation_epoch_end hook
        self.validation_step_outputs = []
        self.alias = alias

    def _create_windows(self, batch, step, window_size, w_idxs=None):
        # Parse common data
        temporal_cols = batch["temporal_cols"]
        temporal = batch["temporal"]

        if step == "train":
            if self.val_size + self.test_size > 0:
                cutoff = -self.val_size - self.test_size
                temporal = temporal[:, :, :cutoff]

            temporal = self.padder_train(temporal)
            if temporal.shape[-1] < window_size:
                raise Exception(
                    "Time series is too short for training, consider setting a smaller input size or set start_padding_enabled=True"
                )
            windows = temporal.unfold(
                dimension=-1, size=window_size, step=self.step_size
            )

            # [B, C, Ws, L+H] 0, 1, 2, 3
            # -> [B * Ws, L+H, C] 0, 2, 3, 1
            windows_per_serie = windows.shape[2]
            windows = windows.permute(0, 2, 3, 1).contiguous()
            windows = windows.reshape(-1, window_size, len(temporal_cols))

            # Sample and Available conditions
            available_idx = temporal_cols.get_loc("available_mask")
            available_condition = windows[:, : self.input_size, available_idx]
            available_condition = torch.sum(available_condition, axis=1)
            final_condition = available_condition > 0
            if self.h > 0:
                sample_condition = windows[:, self.input_size :, available_idx]
                sample_condition = torch.sum(sample_condition, axis=1)
                final_condition = (sample_condition > 0) & (available_condition > 0)
            windows = windows[final_condition]

            # Parse Static data to match windows
            # [B, S_in] -> [B, Ws, S_in] -> [B*Ws, S_in]
            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)
            if static is not None:
                static = torch.repeat_interleave(
                    static, repeats=windows_per_serie, dim=0
                )
                static = static[final_condition]

            # Protection of empty windows
            if final_condition.sum() == 0:
                raise Exception("No windows available for training")

            # Sample windows
            n_windows = len(windows)
            if self.windows_batch_size is not None:
                w_idxs = np.random.choice(
                    n_windows,
                    size=self.windows_batch_size,
                    replace=(n_windows < self.windows_batch_size),
                )
                windows = windows[w_idxs]

                if static is not None:
                    static = static[w_idxs]

            # think about interaction available * sample mask
            # [B, C, Ws, L+H]
            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )
            return windows_batch

        elif step in ["predict", "val"]:

            if step == "predict":
                initial_input = temporal.shape[-1] - self.test_size
                if (
                    initial_input <= self.input_size
                ):  # There is not enough data to predict first timestamp
                    padder_left = nn.ConstantPad1d(
                        padding=(self.input_size - initial_input, 0), value=0
                    )
                    temporal = padder_left(temporal)
                predict_step_size = self.predict_step_size
                cutoff = -self.input_size - self.test_size
                temporal = temporal[:, :, cutoff:]

                #---added by willa!!!
                if self.h < self.patch_len:
                    diff = self.patch_len - self.h
                    padder_right = nn.ConstantPad1d(padding=(0, diff), value=0)
                    temporal = padder_right(temporal)

            elif step == "val":
                predict_step_size = self.step_size
                cutoff = -self.input_size - self.val_size - self.test_size
                if self.test_size > 0:
                    temporal = batch["temporal"][:, :, cutoff : -self.test_size]
                else:
                    temporal = batch["temporal"][:, :, cutoff:]
                if temporal.shape[-1] < window_size:
                    initial_input = temporal.shape[-1] - self.val_size
                    padder_left = nn.ConstantPad1d(
                        padding=(self.input_size - initial_input, 0), value=0
                    )
                    temporal = padder_left(temporal) 

                #---added by willa!!!
                if self.h < self.patch_len:
                    diff = self.patch_len - self.h
                    padder_right = nn.ConstantPad1d(padding=(0, diff), value=0)
                    temporal = padder_right(temporal)

            if (
                (step == "predict")
                and (self.test_size == 0)
                and (len(self.futr_exog_list) == 0)
            ):
                padder_right = nn.ConstantPad1d(padding=(0, self.h), value=0)
                temporal = padder_right(temporal)

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=predict_step_size
            )

            # [batch, channels, windows, window_size] 0, 1, 2, 3
            # -> [batch * windows, window_size, channels] 0, 2, 3, 1
            windows_per_serie = windows.shape[2]
            windows = windows.permute(0, 2, 3, 1).contiguous()
            windows = windows.reshape(-1, window_size, len(temporal_cols))

            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)
            if static is not None:
                static = torch.repeat_interleave(
                    static, repeats=windows_per_serie, dim=0
                )

            # Sample windows for batched prediction
            if w_idxs is not None:
                windows = windows[w_idxs]
                if static is not None:
                    static = static[w_idxs]

            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )
            return windows_batch
        else:
            raise ValueError(f"Unknown step {step}")

    def _normalization(self, windows, y_idx):
        # windows are already filtered by train/validation/test
        # from the `create_windows_method` nor leakage risk
        temporal = windows["temporal"]  # B, L+H, C
        temporal_cols = windows["temporal_cols"].copy()  # B, L+H, C

        # To avoid leakage uses only the lags
        # temporal_data_cols = temporal_cols.drop('available_mask').tolist()
        temporal_data_cols = self._get_temporal_exogenous_cols(
            temporal_cols=temporal_cols
        )
        temporal_idxs = get_indexer_raise_missing(temporal_cols, temporal_data_cols)
        temporal_idxs = np.append(y_idx, temporal_idxs)
        temporal_data = temporal[:, :, temporal_idxs]
        temporal_mask = temporal[:, :, temporal_cols.get_loc("available_mask")].clone()
        if self.h > 0:
            temporal_mask[:, -self.h :] = 0.0

        # Normalize. self.scaler stores the shift and scale for inverse transform
        temporal_mask = temporal_mask.unsqueeze(
            -1
        )  # Add channel dimension for scaler.transform.
        temporal_data = self.scaler.transform(x=temporal_data, mask=temporal_mask)

        # Replace values in windows dict
        temporal[:, :, temporal_idxs] = temporal_data
        windows["temporal"] = temporal

        return windows

    def _inv_normalization(self, y_hat, temporal_cols, y_idx):
        # Receives window predictions [B, H, output]
        # Broadcasts outputs and inverts normalization

        # Add C dimension
        if y_hat.ndim == 2:
            remove_dimension = True
            y_hat = y_hat.unsqueeze(-1)
        else:
            remove_dimension = False

        y_scale = self.scaler.x_scale[:, :, [y_idx]]
        y_loc = self.scaler.x_shift[:, :, [y_idx]]

        y_scale = torch.repeat_interleave(y_scale, repeats=y_hat.shape[-1], dim=-1).to(
            y_hat.device
        )
        y_loc = torch.repeat_interleave(y_loc, repeats=y_hat.shape[-1], dim=-1).to(
            y_hat.device
        )

        y_hat = self.scaler.inverse_transform(z=y_hat, x_scale=y_scale, x_shift=y_loc)
        y_loc = y_loc.to(y_hat.device)
        y_scale = y_scale.to(y_hat.device)

        if remove_dimension:
            y_hat = y_hat.squeeze(-1)
            y_loc = y_loc.squeeze(-1)
            y_scale = y_scale.squeeze(-1)

        return y_hat, y_loc, y_scale

    def _parse_windows(self, batch, windows):
        # Filter insample lags from outsample horizon
        y_idx = batch["y_idx"]
        mask_idx = batch["temporal_cols"].get_loc("available_mask")

        insample_y = windows["temporal"][:, : self.input_size, y_idx]
        insample_mask = windows["temporal"][:, : self.input_size, mask_idx]

        # Declare additional information
        outsample_y = None
        outsample_mask = None
        hist_exog = None
        futr_exog = None
        stat_exog = None

        if self.h > 0:
            outsample_y = windows["temporal"][:, self.input_size :, y_idx]
            outsample_mask = windows["temporal"][:, self.input_size :, mask_idx]

        if len(self.hist_exog_list):
            hist_exog_idx = get_indexer_raise_missing(
                windows["temporal_cols"], self.hist_exog_list
            )
            hist_exog = windows["temporal"][:, : self.input_size, hist_exog_idx]

        if len(self.futr_exog_list):
            futr_exog_idx = get_indexer_raise_missing(
                windows["temporal_cols"], self.futr_exog_list
            )
            futr_exog = windows["temporal"][:, :, futr_exog_idx]

        if len(self.stat_exog_list):
            static_idx = get_indexer_raise_missing(
                windows["static_cols"], self.stat_exog_list
            )
            stat_exog = windows["static"][:, static_idx]

        # TODO: think a better way of removing insample_y features
        if self.exclude_insample_y:
            insample_y = insample_y * 0

        return (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        )

    def training_step(self, batch, batch_idx):
        # Create and normalize windows [Ws, L+H, C]
        window_size = self.input_size + self.patch_len
        windows = self._create_windows(batch, window_size=window_size, step="train")
        y_idx = batch["y_idx"]
        #original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
        original_outsample_y = torch.clone(windows["temporal"][:, -self.patch_len :, y_idx])
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L]
            insample_mask=insample_mask,  # [Ws, L]
            futr_exog=futr_exog,  # [Ws, L + h, F]
            hist_exog=hist_exog,  # [Ws, L, X]
            stat_exog=stat_exog,
        )  # [Ws, S]

        # Model Predictions
        output = self(windows_batch)
        
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=batch["temporal_cols"], y_idx=y_idx
            )
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(y=outsample_y, y_hat=output, mask=outsample_mask)

        if torch.isnan(loss):
            print("Model Parameters", self.hparams)
            print("insample_y", torch.isnan(insample_y).sum())
            print("outsample_y", torch.isnan(outsample_y).sum())
            print("output", torch.isnan(output).sum())
            raise Exception("Loss is NaN, training stopped.")

        self.log(
            "train_loss",
            loss.item(),
            batch_size=outsample_y.size(0),
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, loss.item()))
        return loss

    def _compute_valid_loss(
        self, outsample_y, output, outsample_mask, temporal_cols, y_idx
    ):
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=outsample_y, temporal_cols=temporal_cols, y_idx=y_idx
            )
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            _, sample_mean, quants = self.loss.sample(distr_args=distr_args)

            if str(type(self.valid_loss)) in [
                "<class 'neuralforecast.losses.pytorch.sCRPS'>",
                "<class 'neuralforecast.losses.pytorch.MQLoss'>",
            ]:
                output = quants
            elif str(type(self.valid_loss)) in [
                "<class 'neuralforecast.losses.pytorch.relMSE'>"
            ]:
                output = torch.unsqueeze(sample_mean, dim=-1)  # [N,H,1] -> [N,H]

        # Validation Loss evaluation
        if self.valid_loss.is_distribution_output:
            valid_loss = self.valid_loss(
                y=outsample_y, distr_args=distr_args, mask=outsample_mask
            )
        else:
            output, _, _ = self._inv_normalization(
                y_hat=output, temporal_cols=temporal_cols, y_idx=y_idx
            )
            valid_loss = self.valid_loss(
                y=outsample_y, y_hat=output, mask=outsample_mask
            )
        return valid_loss

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        # TODO: Hack to compute number of windows
        window_size = self.input_size + self.patch_len
        windows_h = self._create_windows(batch, window_size=self.input_size + self.h, step="val")
        n_windows = len(windows_h["temporal"])
        y_idx = batch["y_idx"]

        print(windows_h['temporal'].size())

        # Calculate how many times each window should be repeated
        n_repeats = int(torch.ceil(torch.tensor([self.h/self.patch_len])))
        # Repeat each index in the range of n_windows, n_repeats times
        repeated_idxs = torch.arange(n_windows).repeat_interleave(n_repeats)
        
        valid_losses = []
        batch_sizes = []
        previous_preds = []
        for fi, i in enumerate(repeated_idxs):
            # Create and normalize windows [Ws, L+H, C]
            windows = self._create_windows(batch, window_size=window_size, 
                                           step="val", w_idxs=i)
            windows['temporal'] = windows['temporal'].unsqueeze(0)
            windows = self._normalization(windows=windows, y_idx=y_idx)

            #print(windows['temporal'].size())
            # Parse windows
            (
                insample_y,
                insample_mask,
                _,
                _,
                hist_exog,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)

            # Concatenate along dimension 1
            if fi % n_repeats != 0:
                insample_y = torch.cat((insample_y, *previous_preds), dim=1)
                insample_y = insample_y[:, self.patch_len*len(previous_preds) :]
            else:
                previous_preds = []

            windows_batch = dict(
                insample_y=insample_y,  # [Ws, L]
                insample_mask=insample_mask,  # [Ws, L]
                futr_exog=futr_exog,  # [Ws, L + h, F]
                hist_exog=hist_exog,  # [Ws, L, X]
                stat_exog=stat_exog,
            )  # [Ws, S]

            # Model Predictions
            output = self(windows_batch)
            previous_preds.append(output)

            if (fi+1) % n_repeats == 0:
                output_horizon = torch.cat(previous_preds, dim=1)
                output_horizon = output_horizon[:, : self.h]
                windows_h = self._create_windows(batch, window_size=self.input_size + self.h, 
                                                 step="val", w_idxs=i)
                windows_h['temporal'] = windows_h['temporal'].unsqueeze(0)
                original_outsample_y = torch.clone(windows_h["temporal"][:, -self.h :, y_idx])
                (
                insample_y,
                insample_mask,
                _,
                outsample_mask,
                hist_exog,
                futr_exog,
                stat_exog,
                ) = self._parse_windows(batch, windows_h)
                
                valid_loss_batch = self._compute_valid_loss(
                    outsample_y=original_outsample_y,
                    output=output_horizon,
                    outsample_mask=outsample_mask,
                    temporal_cols=batch["temporal_cols"],
                    y_idx=batch["y_idx"],
                )
                valid_losses.append(valid_loss_batch)
                batch_sizes.append(len(output))
            else:
                continue

        valid_loss = torch.mean(torch.stack(valid_losses))
        batch_sizes = torch.tensor(batch_sizes, device=valid_loss.device)
        batch_size = torch.sum(batch_sizes)

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log(
            "valid_loss",
            valid_loss.item(),
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
        )
        self.validation_step_outputs.append(valid_loss)
        return valid_loss

    def predict_step(self, batch, batch_idx):

        # TODO: Hack to compute number of windows
        window_size = self.input_size + self.patch_len
        windows_h = self._create_windows(batch, window_size=self.input_size + self.h, step="predict")
        # if self.h >= self.patch_len:
        #     n_windows = len(windows["temporal"])-self.patch_len # adjust so windows matches windows_h
        # else:
        #     n_windows = len(windows["temporal"])
        n_windows = len(windows_h["temporal"])
        y_idx = batch["y_idx"]

        # Calculate how many times each window should be repeated
        n_repeats = int(torch.ceil(torch.tensor([self.h/self.patch_len])))
        # Repeat each index in the range of n_windows, n_repeats times
        repeated_idxs = torch.arange(n_windows).repeat_interleave(n_repeats)

        previous_preds = []
        y_hats = []
        for fi, i in enumerate(repeated_idxs):
            # Create and normalize windows [Ws, L+H, C]
            windows = self._create_windows(batch, window_size=window_size, 
                                           step="val", w_idxs=i)

            windows['temporal'] = windows['temporal'].unsqueeze(0)
            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            (
                insample_y,
                insample_mask,
                _,
                _,
                hist_exog,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)

            # Concatenate along dimension 1
            if fi % n_repeats != 0:
                insample_y = torch.cat((insample_y, *previous_preds), dim=1)
                insample_y = insample_y[:, self.patch_len*len(previous_preds) :]
            else:
                previous_preds = []

            windows_batch = dict(
                insample_y=insample_y,  # [Ws, L]
                insample_mask=insample_mask,  # [Ws, L]
                futr_exog=futr_exog,  # [Ws, L + h, F]
                hist_exog=hist_exog,  # [Ws, L, X]
                stat_exog=stat_exog,
            )  # [Ws, S]

            # Model Predictions
            output = self(windows_batch)
            previous_preds.append(output)

            if (fi+1) % n_repeats == 0:
                output_horizon = torch.cat(previous_preds, dim=1)
                output_horizon = output_horizon[:, : self.h]
                windows_h = self._create_windows(batch, window_size=self.input_size + self.h, 
                                                 step="predict", w_idxs=i)
                windows_h['temporal'] = windows_h['temporal'].unsqueeze(0)
                original_outsample_y = torch.clone(windows_h["temporal"][:, -self.h :, y_idx])
                (
                insample_y,
                insample_mask,
                _,
                outsample_mask,
                hist_exog,
                futr_exog,
                stat_exog,
                ) = self._parse_windows(batch, windows_h)

                y_hats.append(self._get_predictions(
                    batch=batch,
                    insample_y=insample_y,
                    output_batch=output_horizon,
                    y_idx=batch["y_idx"],
                    )
                )
            else:
                continue

        y_hat = torch.cat(y_hats, dim=0)
        return y_hat

    def _get_predictions(self, batch, insample_y, output_batch, y_idx):
        # Inverse normalization and sampling
        if self.loss.is_distribution_output:
            _, y_loc, y_scale = self._inv_normalization(
                y_hat=torch.empty(
                    size=(insample_y.shape[0], self.h),
                    dtype=output_batch[0].dtype,
                    device=output_batch[0].device,
                ),
                temporal_cols=batch["temporal_cols"],
                y_idx=y_idx,
            )
            distr_args = self.loss.scale_decouple(
                output=output_batch, loc=y_loc, scale=y_scale
            )
            _, sample_mean, quants = self.loss.sample(distr_args=distr_args)
            y_hat = torch.concat((sample_mean, quants), axis=2)

            if self.loss.return_params:
                distr_args = torch.stack(distr_args, dim=-1)
                distr_args = torch.reshape(
                    distr_args, (len(windows["temporal"]), self.h, -1)
                )
                y_hat = torch.concat((y_hat, distr_args), axis=2)
        else:
            y_hat, _, _ = self._inv_normalization(
                y_hat=output_batch,
                temporal_cols=batch["temporal_cols"],
                y_idx=y_idx,
            )
        return y_hat

    def fit(
        self,
        dataset,
        val_size=0,
        test_size=0,
        random_seed=None,
        distributed_config=None,
    ):
        """Fit.

        The `fit` method, optimizes the neural network's weights using the
        initialization parameters (`learning_rate`, `windows_batch_size`, ...)
        and the `loss` function as defined during the initialization.
        Within `fit` we use a PyTorch Lightning `Trainer` that
        inherits the initialization's `self.trainer_kwargs`, to customize
        its inputs, see [PL's trainer arguments](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

        The method is designed to be compatible with SKLearn-like classes
        and in particular to be compatible with the StatsForecast library.

        By default the `model` is not saving training checkpoints to protect
        disk memory, to get them change `enable_checkpointing=True` in `__init__`.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset`, see [documentation](https://nixtla.github.io/neuralforecast/tsdataset.html).<br>
        `val_size`: int, validation size for temporal cross-validation.<br>
        `random_seed`: int=None, random_seed for pytorch initializer and numpy generators, overwrites model.__init__'s.<br>
        `test_size`: int, test size for temporal cross-validation.<br>
        """
        return self._fit(
            dataset=dataset,
            batch_size=self.batch_size,
            valid_batch_size=self.valid_batch_size,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            distributed_config=distributed_config,
        )

    def predict(
        self,
        dataset,
        test_size=None,
        step_size=1,
        random_seed=None,
        **data_module_kwargs,
    ):
        """Predict.

        Neural network prediction with PL's `Trainer` execution of `predict_step`.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset`, see [documentation](https://nixtla.github.io/neuralforecast/tsdataset.html).<br>
        `test_size`: int=None, test size for temporal cross-validation.<br>
        `step_size`: int=1, Step size between each window.<br>
        `random_seed`: int=None, random_seed for pytorch initializer and numpy generators, overwrites model.__init__'s.<br>
        `**data_module_kwargs`: PL's TimeSeriesDataModule args, see [documentation](https://pytorch-lightning.readthedocs.io/en/1.6.1/extensions/datamodules.html#using-a-datamodule).
        """
        self._check_exog(dataset)
        self._restart_seed(random_seed)
        data_module_kwargs = self._set_quantile_for_iqloss(**data_module_kwargs)

        self.predict_step_size = step_size
        self.decompose_forecast = False
        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            valid_batch_size=self.valid_batch_size,
            **data_module_kwargs,
        )

        # Protect when case of multiple gpu. PL does not support return preds with multiple gpu.
        pred_trainer_kwargs = self.trainer_kwargs.copy()
        if (pred_trainer_kwargs.get("accelerator", None) == "gpu") and (
            torch.cuda.device_count() > 1
        ):
            pred_trainer_kwargs["devices"] = [0]

        trainer = pl.Trainer(**pred_trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        fcsts = torch.vstack(fcsts).numpy().flatten()
        fcsts = fcsts.reshape(-1, len(self.loss.output_names))
        return fcsts

    def decompose(self, dataset, step_size=1, random_seed=None, **data_module_kwargs):
        """Decompose Predictions.

        Decompose the predictions through the network's layers.
        Available methods are `ESRNN`, `NHITS`, `NBEATS`, and `NBEATSx`.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset`, see [documentation here](https://nixtla.github.io/neuralforecast/tsdataset.html).<br>
        `step_size`: int=1, step size between each window of temporal data.<br>
        `**data_module_kwargs`: PL's TimeSeriesDataModule args, see [documentation](https://pytorch-lightning.readthedocs.io/en/1.6.1/extensions/datamodules.html#using-a-datamodule).
        """
        # Restart random seed
        if random_seed is None:
            random_seed = self.random_seed
        torch.manual_seed(random_seed)
        data_module_kwargs = self._set_quantile_for_iqloss(**data_module_kwargs)

        self.predict_step_size = step_size
        self.decompose_forecast = True
        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            valid_batch_size=self.valid_batch_size,
            **data_module_kwargs,
        )
        trainer = pl.Trainer(**self.trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        self.decompose_forecast = False  # Default decomposition back to false
        return torch.vstack(fcsts).numpy()
