import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
        self, 
        channels=None,
        n_classes=10,
        activation=torch.nn.ReLU
    ):
        super().__init__()
        
        layers = []
        
        in_features = channels[0]
        for out_features in channels[1:]:
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(activation())
            in_features = out_features
        
        layers.append(torch.nn.Linear(in_features, n_classes))
        self.net = torch.nn.Sequential(*layers)
        
        
    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.net(signal)
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)
        
        # Put the result into the batch
        batch['signals'] = {'output': signal}
        
        # Perform postprocessing after we get the output
        self.postprocessing(batch)
        
        return batch
    
    def postprocessing(self, batch):
        
        # Take network's output from the batch
        signal = batch['signals']['output']
        
        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal.argmax(dim=1)}