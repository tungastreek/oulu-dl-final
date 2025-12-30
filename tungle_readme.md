# Deep Learning Final Project – Retina Disease Classification (ODIR)

This repository contains the final project implementation for the **Deep Learning** course at the **University of Oulu**.

The goal of this project is **multi-label retinal disease classification** (D: Diabetic Retinopathy, G: Glaucoma, A: AMD) using deep learning models.

Author: **Tung Le**

---

## 1. Environment & Requirements

The environment requirements are **identical to the official course code template**.

If you can run the provided template, you can run this project without installing additional dependencies.

---

## 2. Project Files

The project reuses the original course structure (CSV files, image folders, pretrained checkpoints) and adds:

- `tungle.py`  
  Main training, evaluation, and prediction pipeline.

- `tungle.json`  
  Optional configuration file for overriding default parameters.

---

## 3. Running the Code

The program can be executed **with or without a JSON config file**.

### Using a config file
```bash
python tungle.py --config tungle.json
```

### Using default configuration in code
```bash
python tungle.py
```

All runtime behavior is controlled by the `RunnerConfig` dataclass inside `tungle.py`.

---

## 4. Configurations

Below are some added important configurations and their meaning
- `backbone` and `pretrained_backbone`: Model type and the pretrained backbone state to be loaded. These two must match the same architecture (e.g. `resnet18` cannot load an EfficientNet checkpoint).


- `do_train`, `do_test`, `do_predict`:
  + Boolean values. Control whether to train the model, evaluate on the offsite test set, and generate predictions for the onsite test set.
  + Can be used jointly. For example:
    - Train and evaluate on offsite test set: `do_train = true`, `do_test = true`, `do_predict = false`
    - Evaluate pretrained model on offsite test set and generate submission: `do_train = false`, `do_test = true`, `do_predict = true`
    - Generate submission only: `do_train = false`, `do_test = false`, `do_predict = true`
  + Important: When `do_train = false`, a trained checkpoint must be provided via `pretrained_backbone`.


- `tuning_method`:
  + Has two possible values, `classifier_only` and `full`.
  + Set to `classifier_only` to tune only the classifier head of the model.
  + Set to `full` to tune all parameters.

- `loss_function`: 
  + Controls which loss function to use.
  + Three possible values: `bce`, `focal`, `wbce`.
  + Defaults to `bce`. `focal` and `wbce` are for task 2.

- `attention`: 
  + Controls which attention machanism to use.
  + Three possible values: `none`, `se`, `mha`. These are for task 3.
  + **Only supported on resnet18**
  + Set to "none" to disable

- `label_smoothing_epsilon`:
  + Controls the epsilon value used for label smoothing. This is for task 4.
  + Set to 0 to disable label smoothing

- `epochs`: Number of epochs to run in training mode. Default is very large (1e4) and training relies on early stopping.

- `early_stopping_patience`: Number of epochs without improvement before stopping training.

- `lr`, `weight_decay`: Learning rate and weight decay for optimization. Set `weight_decay` to 0 to disable.

- `save_dir`: Directory to save trained models.

- `predict_output_csv`: Output CSV file when `do_predict = true`

---

## 5. Task-wise Execution Guide

This section describes **how to partial set config for each task**, not how to reproduce exact scores.

If you have trouble running the code, please contact me at **tung.le@student.oulu.fi**

Due to the small dataset size, training variance can be high.  

### Task 1-1 Baseline evaluation
```json
{
    "backbone": "resnet18",
    "do_train": false,
    "do_test": true,
    "do_predict": true,
    "pretrained_backbone": "./pretrained_backbone/ckpt_resnet18_ep50.pt",
    // other config values
}
```

```json
{
    "backbone": "efficientnet",
    "do_train": false,
    "do_test": true,
    "do_predict": true,
    "pretrained_backbone": "./pretrained_backbone/ckpt_efficientnet_ep50.pt",
    // other config values
}
```

### Task 1-2 Baseline classifier-tuning

Example: resnet18
```json
{
    "backbone": "resnet18",
    "tuning_method": "classifier_only",
    "loss_function": "bce",
    "attention": "none",
    "do_train": true,
    "do_test": true,
    "do_predict": true,
    "pretrained_backbone": "./pretrained_backbone/ckpt_resnet18_ep50.pt",
    // other config values
}
```

Example: infer
```json
{
    "backbone": "resnet18",
    "tuning_method": "classifier_only",
    "loss_function": "bce",
    "attention": "none",
    "do_train": false,
    "do_test": true,
    "do_predict": true,
    "pretrained_backbone": "./checkpoints/tungle_task1-2_resnet18.pt",
    // other config values
}
```

### Task 1-3 Baseline full-tuning

Example: resnet18 with classifier-tuned backbone
```json
{
    "backbone": "resnet18",
    "tuning_method": "full",
    "loss_function": "bce",
    "attention": "none",
    "do_train": true,
    "do_test": true,
    "do_predict": true,
    "pretrained_backbone": "./pretrained_backbone/ckpt_resnet18_ep50.pt",
    // other config values
}
```

### Task 2 – Loss function experiments

Example: Focal loss
```json
{
  "backbone": "resnet18",
  "tuning_method": "full",
  "loss_function": "focal",
  "attention": "none",
  "do_train": true,
  "do_test": true,
  "do_predict": true,
  "pretrained_backbone": "./pretrained_backbone/ckpt_resnet18_ep50.pt",
  // other config values
}
```

---

### Task 3 – Attention mechanisms

Example: SE attention
```json
{
  "backbone": "resnet18",
  "tuning_method": "full",
  "loss_function": "bce",
  "attention": "se",
  "do_train": true,
  "do_test": true,
  "do_predict": true,
  "pretrained_backbone": "./pretrained_backbone/ckpt_resnet18_ep50.pt",
  // other config values
}
```

Important: When `do_train = false`, a trained checkpoint must be provided via `pretrained_backbone`, and it must be trained using the **same attention configuration** as the value specified by `attention`.

---

### Task 4 – Final model (Advanced method)

**Swin Transformer + BCE + Label Smoothing**

```json
{
  "backbone": "swin",
  "tuning_method": "full",
  "loss_function": "bce",
  "attention": "none",
  "label_smoothing_epsilon": 0.05,
  "do_train": true,
  "do_test": true,
  "do_predict": true,
  "pretrained_backbone": null, 
  // other config values
}
```