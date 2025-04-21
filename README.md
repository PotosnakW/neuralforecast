# Global Deep Forecasting with Patient-Specific Pharmacokinetics
____

<u><a href="https://arxiv.org/abs/2309.13135">Paper</a></u>

![Model Plot](method.png)

We propose a novel hybrid global-local architecture and a model-agnostic pharmacokinetic (PK) encoder that informs deep learning models of patient-specific treatment effects, achieving significant accuracy improvements on large-scale simulated and real-world blood glucose datasets.


### Preprocess Data
___

Data preprocessing scripts are located in the `./data_preprocessing` folder. Preprocessed `simglucose` data from the open-source [simglucose repository](https://github.com/jxx123/simglucose) is included in the `./datasets` folder.

Please refer to the **"Data and Code Availability"** section of our paper for information on accessing the OhioT1DM 2018 and 2020 datasets. Access to these datasets requires a **Data Use Agreement (DUA)**.

#### 📦 Run preprocessing:
```bash
cd ./data_preprocessing
python preprocess_ohiot1dm_dataset.py
```
> **Note:** Code to preprocess the OhioT1DM dataset assumes that the `Ohiot1dm` data folder is located in the `neuralforecast` repository.

### Installation
___

1. Clone the `neuralforecast` repository and switch to the `pk_paper_code` branch.  
2. Create a conda environment.  
3. Install the `neuralforecast` package.
4. Install the `statsforecast` package.

#### ⚙️ Setup Instructions:
```bash
# Clone the repo and switch to the correct branch
git clone https://github.com/Nixtla/neuralforecast.git
cd neuralforecast
git checkout pk_paper_code

# Create and activate conda environment
conda env create -n neuralforecast python=3.11.9
conda activate neuralforecast

# Install dependencies
pip install -e .
pip install statsforecast
```

### Train Models
___

1. Navigate to the `train_models` folder and run the training script.

#### 🧠 Run Training Script:
```bash
cd ./train_models
python run_training_scripts.py
```

### Evaluate Models
___

Python scripts to generate the table results and figures are included in the `scripts` folder.

Our study includes **several types of evaluation**:

1. **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)**  
   📄 [`results_table.ipynb`](./scripts/results_table.ipynb)  
   This notebook contains example code to compute MAE and RMSE for each model.
   
2. **Percent Improvement** and **Statistical Significance Tests**  
   📄 [`percent_improvement_analysis.ipynb`](./scripts/percent_improvement_analysis.ipynb)  
   This notebook contains example code to compute the percent improvement of hybrid global-local PK models over baseline models, as well as to perform tests for statistical significance.

3. **True Positive Rate (TPR)** and **False Positive Rate (FPR)**  
   📄 [`critical_event_prediction_analysis.ipynb`](./scripts/critical_event_prediction_analysis.ipynb)  
   This notebook evaluates how well models predict critical blood glucose thresholds within forecast windows.

4. **Time Gain** (in minutes)  
   📄 [`time_gain.ipynb`](./scripts/time_gain.ipynb)  
   This notebook estimates how much earlier models can predict hyperglycemic events (blood glucose ≥ 180 mg/dL).
   
5. **Hybrid PK Model vs Local PK Models**  
   📄 [`model_performance_boxplots.ipynb`](./scripts/model_performance_boxplots.ipynb)  
   This notebook generates boxplots of model forecast errors for the hybrid global-local PK model and the local PK models.
   
6. **Model Computational Complexity**  
   📄 [`computational_complexity_analysis.ipynb`](./scripts/computational_complexity_analysis.ipynb)  
   This notebook contains code to calculate the floating-point operations (FLOPs), number of trainable parameters, inference time, and memory usage.

> **Note:** Plotting requires `matplotlib==3.7.0`

### How to Cite
___

Implementations of models used in our work were obtained from the open-source **Neuralforecast** and **StatsForecast** libraries.

#### 📚 Citations:

```bibtex
@misc{potosnak2025hybridpk,
    author={Willa Potosnak and
            Cristian Challú and
            Kin G. Olivares and
            Keith A. Dufendach and
            Artur Dubrawski},
    title = {Global Deep Forecasting with Patient-Specific Pharmacokinetics},
    year={2025},
    howpublished={Proceedings of the Conference on Health, Inference, and Learning, PMLR},
    url={https://github.com/Nixtla/neuralforecast}
}
```

```bibtex
@misc{olivares2022library_neuralforecast,
    author={Kin G. Olivares and
            Cristian Challú and
            Azul Garza and
            Max Mergenthaler Canseco and
            Artur Dubrawski},
    title = {{NeuralForecast}: User friendly state-of-the-art neural forecasting models.},
    year={2022},
    howpublished={{PyCon} Salt Lake City, Utah, US 2022},
    url={https://github.com/Nixtla/neuralforecast}
}
```

```bibtex
@misc{garza2022statsforecast,
    author={Azul Garza, Max Mergenthaler Canseco, Cristian Challú, Kin G. Olivares},
    title = {{StatsForecast}: Lightning fast forecasting with statistical and econometric models},
    year={2022},
    howpublished={{PyCon} Salt Lake City, Utah, US 2022},
    url={https://github.com/Nixtla/statsforecast}
}
```