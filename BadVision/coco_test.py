import torchvision.datasets as dset
import torchvision.transforms as transforms
from nltk import download

cap = dset.CocoCaptions(
                        annFile = '',root = 'COCO/',
                        transform=transforms.PILToTensor())

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(target)