## Preprocess Data
Preprocessing scripts are located in the `./data_preprocessing` folder. Preprocessed `simglucose` data from the open-source `simglucose` repository [1] is included in the `./data` folder.


## Train Models

To train the models, follow these steps:

1. Install the open-source Neuralforecast library [2]. We implemented our method using a copy of the Neuralforecast repository to leverage their model training framework. Also install the open-source Statsforecast library [3].

```bash
pip install -e .
pip install statsforecast
```



2. Navigate to the `./train_models` folder and run the training script:

```bash
cd ./train_models
python run_training_scripts.py
```

## Evaluate Models
To evaluate the models, open and run the `./evaluate_model.ipynb` notebook.


## References:
1. Jinyu Xie. Simglucose v0.2.1 (2018), 2018. URL https://github.com/jxx123/simglucose

2. Kin G. Olivares, Cristian Chall´u, Federico Garza, Max Mergenthaler Canseco, and Artur Dubrawski. NeuralForecast: User friendly state-of-the-art neural forecasting models. PyCon Salt Lake City, Utah, US 2022, 2022. URL https://github.com/Nixtla/neuralforecast.

3. Federico Garza, Max Mergenthaler Canseco, Cristian Challu, and Kin G. Olivares. StatsForecast: Lightning fast forecasting with statistical and econometric models. PyCon Salt Lake City, Utah, US 2022, 2022. URL https://github.com/Nixtla/statsforecast. 
