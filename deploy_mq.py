from mq_deployer import MQDeployer
from mq_net import MQnet
import torch
from generator import Genrator
import numpy as np
from numpy import linalg as LA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
model = MQnet().to(device)
exsit_model_file = "mqv1_1.pth"
model.load_state_dict(torch.load("./models/" + exsit_model_file))
print("Load model file: ", exsit_model_file)
q = MQDeployer(model)
g = Genrator(10)
# for i in range(100):
#    print("norm \n", LA.norm(g[i][0]))
#    print("det \n", LA.det(g[i][1]))
for i in range(len(g)):
    y = q.forward_pass(g[i][1])
    out = y.view(-1, 4).to("cpu").detach().numpy()
    print("y", out)
    print("y soll", g[i][0])
    print("norm y", LA.norm(out))
