from qm_deployer import QMDeployer
from qm_net import QMnet
import torch
from generator import Genrator
import numpy as np
from numpy import linalg as LA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
model = QMnet().to(device)
exsit_model_file = "qmv1_1.pth"
model.load_state_dict(torch.load("./models/" + exsit_model_file))
print("Load model file: ", exsit_model_file)
q = QMDeployer(model)
g = Genrator(10)
# for i in range(100):
#    print("norm \n", LA.norm(g[i][0]))
#    print("det \n", LA.det(g[i][1]))
for i in range(len(g)):
    y = q.forward_pass(g[i][0])
    out = y.view(3, 3).to("cpu").detach().numpy()
    print("y", out)
    print("y soll", g[i][1])
    print("det y", LA.det(out))
