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
### 모델 평가
학습된 모델을 지정된 데이터셋에서 평가하려면 다음 명령어를 사용하세요.
```
python run.py --path <path> --model-path <model_path>
```
* `<path>`: 상호작용하려는 학습된 모델의 경로입니다.
* `<model_path>`: 모델 상태 파일의 경로입니다.
### 모델 학습
지정된 학습 데이터셋에서 모델을 학습시키고, 학습 반복 횟수를 설정하세요.
```
python run.py train --path <trainset_path> --save-to <model_path> --iters <number_iterations>
```
* `<trainset_path>`: 학습 데이터 파일의 경로입니다 (예: train.json 또는 CSV).
* `<model_path>`: 모델 상태 파일을 저장할 경로입니다.


