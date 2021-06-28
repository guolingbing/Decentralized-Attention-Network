# Decentralized Knowledge Graph Representation Learning

## INSTALLATION

Python >= 3.6
Tensorflow >= 2.1.0
Pandas
Numpy
Scipy
scikit-learn

Please run 

```
pip install -r requirements.txt
```
 
to install required packages.


## RUNNING

1. We provided the best parameter settings for each dataset at the end.

2. For example, you can directly copy one of the command in the file and run it in the shell:

```
python run_entity_alignment.py --input ./data/DBP15K/zh_en/mtranse/0_3/ --operator "+" --layernorm True --gpu 0,
```

which will run entity alignment task on ZH-EN dataset with the best setting.

3. Add the parameter "--openEA True" can run open entity alignment task on the corresponding dataset.


## DATA

1. We split the dataset into "data" and "opendata" (and saved them in corresponding folders). Please use "--openEA True" or "--openEP True" to run decentRL on open tasks.

2. Please use "--input" or "--data_path" to spcify the dataset used for training.


## COMMANDS

### Entity alignment

#### ZH-EN
```
python run_entity_alignment.py --input ./data/DBP15K/zh_en/mtranse/0_3/ --operator "+" --layernorm True --gpu 0
```

#### JA-EN
```
python run_entity_alignment.py --input ./data/DBP15K/ja_en/mtranse/0_3/ --operator "+" --layernorm True --gpu 0
```

#### FR-EN
```
python run_entity_alignment.py --input ./data/DBP15K/fr_en/mtranse/0_3/ --operator "+" --layernorm True --gpu 0
```

### Entity prediction

#### FB15K-237
```
python run_entity_prediction.py --data_path ./data/FB15k-237/ --operator "*" --num_layers 2 --layernorm False --gpu 0 --decoder ComplEx
```
#### WN18RR
```
python run_entity_prediction.py --data_path ./data/WN18RR/ --operator "*" --num_layers 2 --layernorm False --gpu 0 --learning_rate 0.001 --decentRL_dp_rate 0.5 --num_sampled 16384 --decoder DistMult 
```
#### FB15K
```
python run_entity_prediction.py --data_path ./data/FB15k/ --operator "*" --num_layers 2 --layernorm False --gpu 0 --learning_rate 0.003 --decoder_dp_rate 0.3 --decoder ComplEx
```
#### WN18
```
python run_entity_prediction.py --data_path ./data/WN18/ --operator "*" --num_layers 2 --layernorm False --gpu 0 --learning_rate 0.001 --decentRL_dp_rate 0.3 --decoder_dp_rate 0.5 --num_sampled 16384 --decoder ComplEx
```