# WaveGenerator

人工的な波形をGANで作成してみるテスト

## Environments

* python3
* keras
* matplotlib
* pandas
* tqdm
* numpy
* tensorflow

## Run

```
$ python3 visualize.py
$ python3 main_noise.py --train --test
```

## Data

1. Remove ZERO data
2. Normalize
3. Outlier

詳細はmain_noise.pyのX_train_generate()
