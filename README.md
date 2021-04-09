# Deep Learning - Practical 2
*Group: 12*

## Overview

Three different architectures can be found in this repo. Each of them can be run by simply installing the requirements.txt

```bash
python -m pip install -r requirements.txt
```

For the LSTM and GPT2, one first needs to create the dataset

```bash
cd Dataset
python create_dataset.py
cd ..
```

All individual models can be run by going into the respective map and running main.py

GPT2 and the LSTM were fully trained on the Google Colab servers using a Tesla T40 GPU.

## Dependencies

A complete list of dependencies can be found in requirements.txt. Additionally, here is a list:

- https://github.com/klaudia-nazarko/nlg-text-generation (for inspiration and some code)
