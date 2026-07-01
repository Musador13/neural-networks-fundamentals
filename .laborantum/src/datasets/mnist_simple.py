import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        self.data = torchvision.datasets.MNIST(root='~/', train=train, download=True)
        self.X = self.data.data
        self.y = self.data.targets

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        sample = {
            'image': self.X[index].float() / 127.5 - 1.0,
            'label': self.y[index].long()
        }

        return sample 
