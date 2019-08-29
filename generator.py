import numpy as np
from torch.utils import data
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA


class Genrator(data.Dataset):
    def __init__(self, size):
        self.size = size
        self.seed = np.random.rand(self.size, 3)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        a, b, c = self.seed[index]
        r = R.from_euler(
            "zxz", [a * 2 * np.pi, b * np.pi, c * 2 * np.pi], degrees=False
        )
        return r.as_quat(), r.as_dcm()


g = Genrator(100)
# for i in range(100):
#    print("norm \n", LA.norm(g[i][0]))
#    print("det \n", LA.det(g[i][1]))
