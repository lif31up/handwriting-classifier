import typing

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(func: typing.Callable) -> typing.Callable:
  def _(root: str, batch_size: int, shuffle: bool) -> tuple:
    dataset = datasets.ImageFolder(root=root, transform=func())
    return dataset, DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
  # init_dataloader

  return _
# get_dataloader

@get_dataloader
def init_dataloader() -> object:
  return transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
  ])  ## compose
# init_dataloader

def main(start, end):
  trainset, loader = init_dataloader(root="./raw/train/", batch_size=3, shuffle=True)
  for idx in range(start, end):
    x, y = trainset[idx]
    print(x, y)
# main

if __name__ == "__main__": main(1, 20)