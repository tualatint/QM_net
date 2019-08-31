import torch


class MQDeployer:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_pass(self, x):
        x = torch.tensor(x)
        x = x.to(self.device, dtype=torch.float)
        x = x.view(-1, 9)
        y = self.model(x)
        return y
