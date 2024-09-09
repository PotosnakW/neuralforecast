## Preprocess Data
Preprocessing scripts are located in the `./data_preprocessing` folder. Preprocessed `simglucose` data from the open-source `simglucose` repository [1] is included in the `./data` folder.


## Train Models

To train the models, follow these steps:

1. Install the open-source Neuralforecast repository [2]. The repository is included here for convenience.

```bash
pip install -e .
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
