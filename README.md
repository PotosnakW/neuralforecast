# MICA: Multivariate Infini Compressive Attention for Time Series Forecasting
____
![Model Plot](method.png)

We propose MICA (Multivariate Infini Compressive Attention), a memory-efficient attention-based forecasting architecture for multivariate time series. MICA adapts Infini-Attention’s linear attention mechanism from context compression to channel compression, enabling computationally efficient cross-channel architecture component that scales linearly with sequence length and channel count. 

<h4><u>Sections:</u></h4>

1. [Environment Setup](#Environment-Setup)
2. [Download Datasets](#Download-Datasets)
3. [Run Experiments](#Run-Experiments)
4. [Experiment Catalog](#Experiment-Catalog)


## Environment Setup

### 1. Create and Activate Conda Environment
```bash
conda create -n neuralforecast python=3.11.0
conda activate neuralforecast

git clone <anonymous>
cd ./neuralforecast
git checkout remotes/origin/moment_infini
pip install -e .
pip install -r mica_requirements.txt

```


## Download Datasets

### 1. Download GiftEval Datasets

Download the required datasets from the GiftEval benchmark (requires that `git lfs` is installed):
```bash
# Clone the GiftEval dataset from huggingface 
git clone https://huggingface.co/datasets/Salesforce/GiftEval
```


### 2. Download and Preprocess Iowa Windspeed Datasets

For each linked below. Follow the directions to submit data orders:

PLOWS: Iowa Automated Weather Observing System (AWOS) 1-minute Data ([link](https://data.eol.ucar.edu/dataset/113.038))

SMEX02: Automated Weather Observing System (AWOS) Iowa 1-min Data ([link](https://data.eol.ucar.edu/dataset/80.003))

IHOP_2002: Automated Weather Observing System (AWOS) Iowa 1-min Data ([link](https://data.eol.ucar.edu/dataset/77.099))

Change the folder names to `iowa_PLOWS_data`, `iowa_SMEX02_data`, and `iowa_IHOP_data`.


```bash
cd ~/long_context_tsfms/preprocessing

# before running the following scripts
# modify "data_dir=..." variable to point to the downloaded folders
python preprocess_iowa_IHOP_SMEX02_datasets.py
python preprocess_iowa_PLOWS_dataset.py
```


### 3. Preprocess Simglucose Dataset

```bash
git clone <anonymous>
cd ~/simglucose
git checkout remotes/origin/harrison_benedict_eqn
pip install -e .

cd ~/long_context_tsfms/preprocessing
python preprocess_simglucose_dataset.py
```

## Run Experiments

### 1. Configure Experiment Settings

Open `launch_exp.py` in the `training` folder and modify the following parameters:

#### Set Output Directory
```python
SAVE_DIR = "/path/to/your/output/directory"  # Change this to your desired output location
```

#### Set file name
```python
FILE_NAME = "train_models"  # train_models used for all experiments except 'vanilla_pca_t5tiny' and 'vanilla_ica_t5tiny' which use "train_models_pca".
```

#### Set Experiment Name
```python
EXPERIMENT_NAME = "my_experiment_name"  # See experiment names in experiment catalog below
```

#### Set GiftEval dataset location
```python
GIFT_EVAL_DIR = '/path/to/your/downloaded/GiftEval' # This is the folder that contains all GiftEval datasets (see step 1 in Download Datasets section)
```

#### Select Datasets
Comment out datasets you **don't** want to run. Leave uncommented only the datasets you want to experiment with:
```python
# Example: To run only ETTh1 and Weather datasets, comment out the others
datasets = [
    'ETTh1',           # Keep uncommented
    # 'ETTh2',         # Commented out - won't run
    # 'ETTm1',         # Commented out - won't run
    'Weather',         # Keep uncommented
    # 'Electricity',   # Commented out - won't run
]
```

#### Set GPU indices
```python
GPU_INDICES = [0, 1, 2, 3]  # Update this based on your available GPU resources
```

### 2. Launch Experiments
```bash
python3 launch_experiments.py # This file automatically  creates tmux sessions, distributes, and launches the experiments among gpus based on the dataset and random_seed combinations.
```



## Experiment Catalog

Experiment names are shown in parantheses. Use these names for the EXPERIMENT_NAME parameter in `launch_exp.py`.

### Infini-Attention Variants

#### 1. Infini MLP-Mixer (`infini_mlpmixer_t5tiny`)
Infini-attention with MLP-based gating:
- With channel exclusion
- Without channel exclusion

#### 2. Infini MLP-Query-Mixer (`infini_mlpquerymixer_t5tiny`)
Enhanced MLP gating incorporating query information:
- With channel exclusion
- Without channel exclusion

#### 3. Infini Layer-Wise (`infini_layerwise_t5tiny`)
Beta gating with layer-specific parameters:
- With channel exclusion
- Without channel exclusion

#### 4. Infini Layer-Wise Channel-Wise (`infini_layerwise_channelwise_t5tiny`)
Beta gating with layer and channel-specific parameters:
- With channel exclusion
- Without channel exclusion

#### 5. Infini Channel-Wise (`infini_channelwise_t5tiny`)
Simplified gating with shared beta across channels:
- With channel exclusion
- Without channel exclusion

#### 6. Infini Shared Beta (`infini_t5tiny`)
Simplified gating with shared beta across layers and channels:
- With channel exclusion
- Without channel exclusion

### Baseline Models

#### 7. Vanilla T5-Tiny (`vanilla_t5tiny`)
Standard attention baseline:
- Univariate head implementation
- Multivariate head implementation

#### 8. Vanilla T5-Tiny with PCA (`vanilla_pca_t5tiny`)
- Vanilla attention with PCA preprocessing (univariate head)


#### 9. Multivariate MLP (`multivariateMLP_baseline`)
- Univariate mode (first dimension: `n_channels * windows_batch_size`)
- Multivariate mode (first dimension: `windows_batch_size`, last dimension: `n_channels`)

#### 10. TSMixer (`tsmixer_baseline`)
- Univariate mode
- Multivariate mode

#### 11. iTransformer (`itransformer_baseline`)
Standard and T5-based variants:
- iTransformer univariate
- iTransformer multivariate
- iTransformerT5 univariate
- iTransformerT5 multivariate

#### 12. Timer-XL (`timerxl_baseline`)
- Univariate mode
- Multivariate mode

#### 13. Crossformer (`crossformer_baseline`)
- Univariate mode
- Multivariate mode

#### 14. AutoETS (`AutoETS`)