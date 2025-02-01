from tqdm import tqdm
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from model.ConvNet import ConvNet

def main(path: str, save_to: str, iters: int):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device
  # create Dataset
  transformer = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path, transform=transformer)
  loader = DataLoader(imageset, shuffle=True, batch_size=4)

  # define ConvNet
  n_classes = len(imageset.classes)
  model = ConvNet(n_classes=n_classes).to(device=device)
  criterion = torch.nn.CrossEntropyLoss()
  lr, weight_decay = 0.001, 0.01
  optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  # training algroithm
  for _ in tqdm(range(iters)):
    for x, y in loader:
      loss = criterion(model.forward(x), y)
      optim.zero_grad()
      loss.backward()
      optim.step()
  # for for
  print(f"loss: {loss.item():.4f}") # print result of training

  # saving the weights
  feature = {
    "state": model.state_dict(),
    "output": n_classes,
    "labels": imageset.classes
  } # feature
  torch.save(feature, save_to)
# main()

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Korean", "./model/model.pth", 5)