`pytorch` `tqdm`
* **task**: classifying handwriting image into its character index.
* **dataset**: downloaded from `pytorch` dataset library.

## Handwriting Classifier using Convolutional Network
This repository contains a **handwriting classification model** implemented using PyTorch. The model is designed to recognize handwritten characters or digits, trained on a dataset such as MNIST or EMNIST. The project is optimized for training on a laptop and can be used for educational purposes or as a starting point for more complex handwriting recognition tasks.

- **CNN-based Model**: A lightweight convolutional neural network for handwriting classification.
- **Easy-to-Use**: Simple scripts for training, evaluation, and inference.
- **Customizable**: Easy to modify the model architecture or dataset for your needs.

---

## Instruction
### Evaluate Model
Use this command to evaluate your trained model on a specified dataset.
```
python run.py --path <path> --model-path <model_path>
```
* `<path>`: Path to the trained model you wish to interact with.
* `<model_path>`: Path to the model state file.
### Train Model
Train your model on a specified training dataset and set the number of iterations for training.
```
python run.py train --path <trainset_path> --save-to <model_path> --iters <number_iterations>
```
* `<trainset_path>`: Path to your training data file (e.g., train.json or CSV).
* `<model_path>`: Path to save the model state file.
* `<number_iters>`: Number of training iterations to run. This controls how many times the model will learn from the data.