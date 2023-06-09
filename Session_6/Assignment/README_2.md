### Assignment Part 2:

We have considered many points in our last 5 lectures. Some of these we have covered directly and some indirectly.
They are:
1. How many layers,
2. MaxPooling,
3. 1x1 Convolutions,
4. 3x3 Convolutions,
5. Receptive Field,
6. SoftMax,
7. Learning Rate,
8. Kernels and how do we decide the number of kernels?
9. Batch Normalization,
10. Image Normalization,
11. Position of MaxPooling,
12. Concept of Transition Layers,
13. Position of Transition Layer,
14. DropOut
15. When do we introduce DropOut, or when do we know we have some overfitting
16. The distance of MaxPooling from Prediction,
17. The distance of Batch Normalization from Prediction,
18. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
19. How do we know our network is not going well, comparatively, very early
20. Batch Size, and Effects of batch size,etc (you can add more if we missed it here)

1. Refer to this code: [COLABLINK](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx)
#### WRITE IT AGAIN SUCH THAT IT ACHIEVES:
* 99.4% validation accuracy
* Less than 20k Parameters
* You can use anything from above you want. 
* Less than 20 Epochs
* Have used BN, Dropout,
* (Optional): a Fully connected layer, have used GAP. 

2. To learn how to add different things we covered in this session, you can refer to this code: (https://www.kaggle.com/code/enwei26/mnist-digits-pytorch-cnn-99/notebook) DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.
3. This is a slightly time-consuming assignment, please make sure you start early. You are going to spend a lot of effort running the programs multiple times
Once you are done, submit your results in S6-Assignment-Solution
4. You must upload your assignment to a public GitHub Repository. Create a folder called S6 in it, and add your iPynb code to it. THE LOGS MUST BE VISIBLE. Before adding the link to the submission make sure you have opened the file in an "incognito" window. 
5. If you misrepresent your answers, you will be awarded -100% of the score.
6. If you submit a Colab Link instead of the notebook uploaded on GitHub or redirect the GitHub page to Colab, you will be awarded -50%
Submit details to S6 - Assignment QnA. 

### CODE WALKTHROUGH

Code block 1:

* importing libraries
``` python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
```
Code block 2:

* Data transformations
```python
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
```
Code block 3:

* Preparing Train_loader and test_loader, assigning batch size = 64
```python
train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=transforms)

batch_size = 64

kwargs = {'batch_size': batch_size, 'shuffle': True,  'num_workers': 2, 'pin_memory': True}

# test_loader = torch.utils.data.DataLoader(train_data, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
```
Code block 4:

* Data Visualization
```python
import matplotlib.pyplot as plt

batch_data, batch_label = next(iter(train_loader)) 

fig = plt.figure()

for i in range(12):
  plt.subplot(3,4,i+1)
  plt.tight_layout()
  plt.imshow(batch_data[i].squeeze(0), cmap='gray')
  plt.title(batch_label[i].item())
  plt.xticks([])
  plt.yticks([])
 ```
<img width="373" alt="image" src="https://github.com/Nishasathish13/TSAI_ERA/assets/75114179/bb9111e5-e8a0-4802-a7dc-2f36c75ad5b1">

Code block 5:

* Model
* Using convolution layers, Batch Normalization, Max Pooling, Dropout(10%) and a GAP layer at the end (No FC layer)
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, bias = False) #input -28 OUtput-26 RF-3
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, bias = False)  #input -26 OUtput-24 RF-5
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) #input -24 OUtput-12 RF-10
        self.conv3 = nn.Conv2d(32, 16, 3, bias = False) #input -12 OUtput-10 RF-12
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, bias = False) #input -10 OUtput-8 RF-14
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) #input -8 OUtput-4 RF-24
        self.conv5 = nn.Conv2d(32, 10, 4, bias = False) # Input - 4, Output - 2, RF  - 30        
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.pool1(self.dropout(F.relu(self.bn2(self.conv2(self.dropout(F.relu(self.bn1(self.conv1(x)))))))))
        #print(x.shape)
        x = self.pool2(self.dropout(F.relu(self.bn4(self.conv4(self.dropout(F.relu(self.bn3(self.conv3(x)))))))))
        #print(x.shape)
        x = self.conv5(x)
        #print(x.shape)
        x = x.view(-1, 10)
        #print(x.shape)
        return F.log_softmax(x)
  ```
  
*The above code gives a total of 19,280 (Trainable) parameters
  
<img width="492" alt="image" src="https://github.com/Nishasathish13/TSAI_ERA/assets/75114179/4e3af8b2-3270-4dbb-a36f-2b753780bbae">

```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 21):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

*Last few epoch logs

<img width="488" alt="image" src="https://github.com/Nishasathish13/TSAI_ERA/assets/75114179/925d8c24-d56d-4dd9-a464-2621d607aeca">


* Highest accuracy achieved = 99.46% in the 20th epoch
* Learning rate = 0.01

loss=0.0020355198066681623 batch_id=937: 100%|██████████| 938/938 [01:25<00:00, 10.97it/s]

Test set: Average loss: 0.0169, Accuracy: 9946/10000 (99.5%)
