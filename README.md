# HYU-AUE8088, Understanding and Utilizing Deep Learning
## PA #1. Image Classification

# Files

```bash
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── metric.py
│   ├── network.py
│   └── util.py
├── test.py
└── train.py
```


# 0. Preparation

### Setup virtual environment
- Create python virtual environment
```bash
$ python3 -m venv venv/aue8088
$ source venv/aue8088/bin/activate
```

- Check whether the virtual environment set properly
: The result should end with `venv/aue8088/bin/python`.

```bash
$ which python
```

- Install required packages
```bash
$ pip install -r requirements.txt
```
### Wandb setup
- Login

```bash
$ wandb login
```

- Specify your Wandb entity
```bash
$ echo "export WANDB_ENTITY={YOUR_WANDB_ENTITY}" >> ~/.bashrc
$ source ~/.bashrc
```

# 1. [TODO] Evaluation metric
### Finish `MyAccuracy` class (src/metric.py)
- Please complete this function to measure accuracy of the prediction

### Implement `MyF1Score` class (src/metric.py)
- Please write MyF1Score class from scratch
    + Calculate per-class F-1 score in a one-vs-rest manner
- Apply this new metric (hint: update src/network.py)


# 2. [TODO] Train models
- Try different settings (src/config.py)

```bash
$ python train.py
```

# 3. [TODO] Toward state-of-the-art
- How to improve performance more?
    + Find state-of-the-art method/model(paper) on TinyImageNet-200 dataset
    + Check difference between baseline and state-of-the-art
    + Apply missing stuff in the baseline
