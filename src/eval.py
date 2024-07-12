from tqdm import tqdm
import torch

from src.model.CNN import CNN
from src.data.transform import *

def main(path: str):
  trainset, loader = init_dataloader("./data/raw/train", 1, True)

  data = torch.load(path)
  state = data["state"]
  labels = data["labels"]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CNN().to(device)
  model.load_state_dict(state)
  model.eval()
# main

if __name__ == "__main__": main("./model/CNN.pth")