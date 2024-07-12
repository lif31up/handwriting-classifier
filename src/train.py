from tqdm import tqdm
import torch

from src.model.CNN import CNN
from src.data.transform import *

def main(path: str, save_to: str, iters: int):
  trainset, loader = init_dataloader(path, 1, True)

  # read to train
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CNN().to(device=device)
  criterion = torch.nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

  # tra!n
  for _ in tqdm(range(iters)):
    for x, y in loader:
      loss = criterion(model.forward(x), y)
      optim.zero_grad()
      loss.backward()
      optim.step()
  # for for
  print(f"loss: {loss.item():.4f}")

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
    "labels": trainset.class_to_idx,
  }  # features
  torch.save(features, save_to)
# main

if __name__ == "__main__": main("./data/raw/test", "./model/CNN.pth", 100)