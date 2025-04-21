# Global Deep Forecasting with Patient-Specific Pharmacokinetics
____

#### 📄 **[Paper (arXiv)](https://arxiv.org/abs/2309.13135)**

![Model Plot](method.png)

We propose a novel hybrid global-local architecture and a model-agnostic pharmacokinetic (PK) encoder that informs deep learning models of patient-specific treatment effects, achieving significant accuracy improvements on large-scale simulated and real-world blood glucose forecasting datasets.

<h4><u>Sections:</u></h4>

1. [Installation](#Installation)
2. [Preprocess Data](#Preprocess-Data)
3. [Train Models](#Train-Models)
4. [Evaluate Models](#Evaluate-Models)
5. [How to Cite](#How-to-Cite)
6. [Contributing](#Contributing)
7. [License](#License)

### Installation
___

1. Clone the [`neuralforecast`](https://github.com/PotosnakW/neuralforecast) repository and switch to the `pk_paper_code` branch.  
2. Create a conda environment.  
3. Install the `neuralforecast` package.
4. Install the [`statsforecast`](https://github.com/Nixtla/statsforecast) package.

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

### Preprocess Data
___

Data preprocessing scripts are located in the `./data_preprocessing` folder. Preprocessed `simglucose` data from the open-source [simglucose repository](https://github.com/jxx123/simglucose), used in our experiments, is included in the [datasets](https://github.com/PotosnakW/neuralforecast/tree/pk_paper_code/datasets) folder.

Please refer to the **"Data and Code Availability"** section of our paper for information on accessing the OhioT1DM 2018 and 2020 datasets. Access to these datasets requires a **Data Use Agreement (DUA)**.

#### 📦 Run preprocessing:
```bash
cd ./data_preprocessing
python preprocess_ohiot1dm_dataset.py
```
> **Note:** The `preprocess_ohiot1dm_dataset.py` file assumes that the `Ohiot1dm` data folder is located in the parent directory, (i.e., '../OhioT1DM').

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

> **Note:** Plot formatting requires `matplotlib==3.7.0`

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
   This notebook contains code to calculate the floating-point operations per second (FLOPs), number of trainable parameters, inference time, and memory usage.

### How to Cite
___

Implementations of models used in our work were obtained from the open-source [**Neuralforecast**](https://github.com/Nixtla/neuralforecast) and [**StatsForecast**](https://github.com/Nixtla/statsforecast) libraries. In addition to citing our paper, please also include citations for the aforementioned libraries.

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

### Contributing
___
Bug reports and pull requests are welcome.

### License
___

MIT License

Copyright (c) 2022 Carnegie Mellon University, [Auton Lab](http://autonlab.org)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<img align="right" height="100px" width="100px" src="./auton_logo.png">
<img align="right" height="70px" width="140px" src="./cmu-wordmark-stacked-r.png">

