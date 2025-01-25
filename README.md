## Preprocess Data
Datasets are located in the `./data` folder. Preprocessing scripts for the synthetic data experiments are located in the `./preprocessing` folder. Synthetic data will automatically be generated when `synthetic_sinusoid_composition` is called. 


## Train Models

To train the models, follow these steps:

1. Install the open-source Neuralforecast library [2]. We implemented our method using a copy of the Neuralforecast repository to leverage their model training framework. Also install the open-source Statsforecast library [3].

```bash
pip install -e .
pip install statsforecast
```


2. Navigate to the `./training` folder. In the `run_train_script.py` file, edit the variables: experiment and experiment_mode to specify which experiment to run. Edit save_dir to specify directory where results will be saved.

4. Run the training script:

```bash
cd ./train_models
python run_train_script.py
```

## Evaluate Models
1. Navigate to the `./evaluation` folder. In the `run_eval_script.py` file, edit the variables: experiment and experiment_mode to specify which experiment to evaluate.

2. To evaluate the models, open and run the `./run_eval_script.py` file.

3. The `table_results.ipynb` notebook can be used to generate results in the main tables and for specific ablation experiments.


## References for open-source code repositories used in this work:
1. Kin G. Olivares, Cristian Chall´u, Federico Garza, Max Mergenthaler Canseco, and Artur Dubrawski. NeuralForecast: User friendly state-of-the-art neural forecasting models. PyCon Salt Lake City, Utah, US 2022, 2022. URL https://github.com/Nixtla/neuralforecast.

2. Federico Garza, Max Mergenthaler Canseco, Cristian Challu, and Kin G. Olivares. StatsForecast: Lightning fast forecasting with statistical and econometric models. PyCon Salt Lake City, Utah, US 2022, 2022. URL https://github.com/Nixtla/statsforecast. 
