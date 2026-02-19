# MICA: Multivariate Infini Compressive Attention for Time Series Forecasting
____

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

![Model Plot](method.png)

We propose MICA (Multivariate Infini Compressive Attention), a memory-efficient attention-based forecasting architecture for multivariate time series. MICA adapts compressive memory techniques with linear attention, originally developed for long-context language models, from context com- pression to channel compression, enabling a computationally efficient cross-channel architecture component that scales linearly in both time and memory with sequence length and channel count.

We implement our method and various baselines within the open-source neuralforecast repository [1], leveraging its available models and standardized training infrastructure for consistent benchmarking.

1. Olivares, K. G., Challú, C., Garza, F., Canseco, M. M., and Dubrawski, A. NeuralForecast: User friendly state-of-the-art neural forecasting models. PyCon Salt Lake City, Utah, US 2022, 2022.

<h4><u>Sections:</u></h4>

1. [Environment Setup](#Environment-Setup)
2. [Download Datasets](#Download-Datasets)
3. [Train Experiment Models and Get Forecasts](#Run-Experiments)
4. [Evaluate Forecasts](#Eval-Fcsts)
5. [Experiment Catalog](#Experiment-Catalog)
6. [Reference](#Reference)



## Environment Setup

### 1. Create and Activate Conda Environment
```bash
conda create -n neuralforecast python=3.11.0
conda activate neuralforecast 

git clone <repo>
cd ./neuralforecast
git checkout remotes/origin/mica
pip install -e .
cd ./mica
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

Change the folder names to `iowa_PLOWS_data`, `iowa_SMEX02_data`, and `iowa_IHOP_data`, respectively. Move these dataset folders to the ./mica folder.


```bash
cd ~/neuralforecast/mica/preprocessing
python preprocess_iowa_IHOP_SMEX02_datasets.py
python preprocess_iowa_PLOWS_dataset.py
```


### 3. Preprocess Simglucose Dataset

```bash
git clone git@github.com:{anon}/simglucose.git
cd ~/simglucose
git checkout remotes/origin/harrison_benedict_eqn
pip install -e .

cd ~/neuralforecast/mica/preprocessing
python preprocess_simglucose_dataset.py
```

## Train Experiment Models and Get Forecasts

### 1. Configure Experiment Settings

Open `launch_exp.py` in the `./mica/training` folder and modify the following parameters:

#### Set Output Directory
```python
SAVE_DIR = "/path/to/your/output/directory"  # Change this to your desired output location
```

#### Set file name
```python
FILE_NAME = "train_models" FILE_NAME = "train_models"  # train_models used for all experiments except 'vanilla_pca_t5tiny' and 'vanilla_ica_t5tiny' which use "train_models_pca" or 'chronos2.0_baselin' which uses "zeroshot_models"
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

After running experiments, results will be saved in your `SAVE_DIR`:
```
SAVE_DIR/
├── {EXPERIMENT_NAME}/
│   ├── {DATASET_NAME}/
│   │   ├── rs1_ishm2_h{HORIZON}/
│   │   │   ├── {model_name}.ckpt
│   │   │   └── forecast.csv
│   │   ├── rs2_ishm2_h{HORIZON}/
│   │   ├── rs3_ishm2_h{HORIZON}/
│   │   ├── rs4_ishm2_h{HORIZON}/
│   │   └── rs5_ishm2_h{HORIZON}/
│   └── ...
```

**Example:**
```
SAVE_DIR/
├── vanilla_t5tiny/
│   ├── ett1_D/
│   │   ├── rs1_ishm2_h96/
│   │   │   ├── AutoMOMENT_vanilla_0.ckpt
│   │   │   ├── AutoMOMENT_vanilla_headmixer_0.ckpt
│   │   │   ├── AutoPatchTSTMultivariate_vanilla_0.ckpt
│   │   │   ├── AutoPatchTSTMultivariate_vanilla_headmixer_0.ckpt
│   │   │   └── forecast.csv
│   │   ├── rs2_ishm2_h96/
│   │   └── ...
│   └── Weather/
│       └── ...
```

**Note:** `rs{1-5}` indicates 5 random seed runs per dataset/horizon combination. Each run contains checkpoints for all model variants tested.


## Evaluate Forecasts
Specify the experiment name and the GiftEval repo path within the command to evaluate forecast results:

```
cd ~/neuralforecast/mica/training/
python -m forecast_error --experiment_name t5tiny_vanilla --GIFT_EVAL_path /home/GiftEval
```


## Experiment Catalog

| Category | Experiment Name | Gating Mechanism | Channel Exclusion | Head Type |
|----------|----------------|------------------|-------------------|-----------|
| **Infini Variants** | `infini_mlpmixer_t5tiny` | MLP-based | ✓ / ✗ | - |
| | `infini_mlpquerymixer_t5tiny` | MLP + Query | ✓ / ✗ | - |
| | `infini_layerwise_t5tiny` | Layer-specific β | ✓ / ✗ | - |
| | `infini_layerwise_channelwise_t5tiny` | Layer + Channel β | ✓ / ✗ | - |
| | `infini_channelwise_t5tiny` | Channel-specific β | ✓ / ✗ | - |
| | `infini_t5tiny` | Shared β | ✓ / ✗ | - |
| **Baselines** | `vanilla_t5tiny` | Standard attention | - | Uni / Multi |
| | `vanilla_pca_t5tiny` | Standard + PCA | - | Univariate |
| | `multivariateMLP_baseline` | - | - | - |
| | `tsmixer_baseline` | - | - | - |
| | `itransformer_baseline` | - | - | - |
| | `timerxl_baseline` | - | - | - |
| | `crossformer_baseline` | - | - | - |
| | `AutoETS` | Statistical | - | - |

**Note:** ✓ = with channel exclusion, ✗ = without channel exclusion
