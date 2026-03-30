# MICA: Multivariate Infini Compressive Attention for Time Series Forecasting
____

## 📋 Rebuttal Materials


**Table 1**: Weighted channels ablation study: Comparison of uniform (U), static (S), and dynamic (D) channel weighting variants against the PatchTST vanilla baseline. Values are MAE averaged over 5 random seeds with standard deviation in parentheses. **Bold** = best, <u>Underlined</u> = second-best.

| **Dataset** | **Freq.** | **Vanilla** | **MICA (U)** | **MICA (S)** | **MICA (D)** |
|---|---|---|---|---|---|
| COVID Deaths | D | 141.437 (43.419) | <u>135.718</u> (40.678) | **130.176** (27.018) | 143.915 (86.827) |
| Jena Weather | H | 9.911 (0.225) | 9.543 (0.298) | <u>9.517</u> (0.176) | **9.476** (0.223) |
| | D | 14.005 (0.185) | **13.799** (0.328) | 13.944 (0.260) | <u>13.889</u> (0.382) |
| M-DENSE | D | 53.723 (0.236) | <u>51.959</u> (0.203) | 52.078 (0.302) | **51.594** (1.182) |
| ETT1 | D | **155.534** (2.508) | <u>157.871</u> (2.663) | 159.448 (2.065) | 158.186 (1.721) |
| | W | 1003.563 (29.109) | <u>959.206</u> (42.454) | **938.360** (22.515) | 965.903 (27.555) |
| ETT2 | D | 260.281 (11.639) | <u>254.152</u> (5.445) | **253.327** (5.868) | 255.058 (9.701) |
| | W | 3016.452 (686.651) | **2249.077** (188.625) | 2436.890 (214.840) | <u>2324.161</u> (157.982) |
| Solar | D | 258.389 (1.663) | **252.718** (1.232) | 254.602 (1.290) | <u>252.852</u> (2.516) |
| | W | 1127.889 (20.369) | **1058.990** (38.140) | 1067.190 (25.051) | <u>1060.330</u> (46.974) |
| **Average Rank** | | 3.6 | **1.7** | 2.4 | 2.3 |

<br>

**Table 2**: TimeMixer Results.


<br>

**Table 3**: Longer context ablation study: Values are MAE averaged over 5 random seeds with standard deviation in parentheses. Blue results indicate lower forecast error of MICA compared with the univariate model counterpart. 
| **Dataset** | **Freq.** | **MOMENT Baseline** | **MOMENT-MICA** | **PatchTST Baseline** | **PatchTST-MICA** |
|---|---|---|---|---|---|
| M-DENSE | D | 48.705 (0.772) | 50.247 (0.765) | 55.009 (0.475) | <span style="color:blue">52.655 (1.624)</span> |
| Jena Weather | H | 10.205 (0.121) | <span style="color:blue">9.536 (0.086)</span> | 10.432 (0.183) | <span style="color:blue">9.930 (0.208)</span> |
| | D | 11.438 (0.249) | 12.890 (0.905) | 12.363 (0.293) | 12.504 (0.133) |
| ETT1 | D | 163.611 (2.754) | <span style="color:blue">158.836 (4.332)</span> | 157.270 (1.417) | 159.723 (1.769) |
| | W | 979.311 (14.820) | 999.862 (22.948) | 935.219 (38.262) | 947.473 (19.065) |
| **Average Rank** | | 2.5 | 2.3 | 2.7 | 2.5 |

<br>

![flops](channel_scale_parameter_impact_figure.png)
**Figure**: GFLOPs and inference speed (ms) as a function of channel count. Chronos-2 scales steeply, reaching approximately 1534 GFLOPs and 296ms inference time at C=600, compared to PatchTST-MICA and MOMENT-MICA which remain below 60 GFLOPs and under 18ms at the same channel count, representing a 25.6x reduction in computational cost and 16.4x faster inference. At C=600, Timer-XL reaches approximately 166 GFLOPs and 188ms inference time, and Crossformer reaches approximately 205 GFLOPs and 50ms inference time, compared to PatchTST-MICA and MOMENT-MICA which remain below 60 GFLOPs and under 18ms, representing a 3x computational and up to 11x inference speed advantage over these baselines.


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
