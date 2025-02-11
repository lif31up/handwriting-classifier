import torch
from torch.utils.data import DataLoader
from src.model.ConvNet import ConvNet
import torchvision as tv

def main(path: str, model: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load the model from path
  saved = torch.load(model)
  model = ConvNet(saved["n_oupt"]).to(device=device)
  model.load_state_dict(saved["state"])
  model.eval()

  # create Dataset and transformer
  transformer = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path, transform=transformer)
  loader = DataLoader(imageset, shuffle=True, batch_size=1)

  # define and calculate loss
  criterion = torch.nn.CrossEntropyLoss()
  loss = float()
  for x, y in loader:
    loss = criterion(model.forward(x), y)
  # for
  print(f"loss: {loss.item():.4f}")  # print result of training
# main()

if __name__ == "__main__": main(path="../data/raw/omniglot-py/images_background/Korean", model="./model/model.pth")