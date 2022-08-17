# Import PyTorch
import torch

# We use torchvision to get our dataset and useful image transformations
import torchvision
import torchvision.transforms as transforms

# Import PyTorch's optimization libary and nn
# nn is used as the basic building block for our Network graphs
import torch.optim as optim
import torch.nn as nn

# Are we using our GPU?

print("GPU available: {}".format(torch.cuda.is_available()))


#If GPU is available set device = 'cuda' if not set device = 'cpu'

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

  # Transform to a PyTorch tensors and the normalize our valeus between -1 and +1
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])

  # Load our Training Data and specify what transform to use when loading
  trainset = torchvision.datasets.MNIST('mnist',
                                        train=True,
                                        download=True,
                                        transform=transform)

  # Load our Test Data and specify what transform to use when loading
  testset = torchvision.datasets.MNIST('mnist',
                                       train=False,
                                       download=True,
                                       transform=transform)