from qm_net import QMnet
from generator import Genrator
from torch.utils import data
import torch
import numpy as np
from qm_trainer import QMTrainer

params = {"batch_size": 100, "shuffle": True, "num_workers": 6}
data_size = np.int64(1e8)
training_set = Genrator(data_size)
print("training dataset size:", len(training_set))
training_generator = data.DataLoader(training_set, **params)
validation_generator = data.DataLoader(training_set, **params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

learning_rate = 1e-4

exp_index = 1
model = QMnet().to(device)
exsit_model_file = "qmv1_1.pth"
model.load_state_dict(torch.load("./models/" + exsit_model_file))
print("Load model file: ", exsit_model_file)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

trainer = QMTrainer(
    model,  # network model
    criterion,  # loss function
    optimizer,
    training_generator,  # training data wrapper
    validation_generator,  # validation data wrapper
    exp_index,  # experiment serial number
    max_epoch=100,  # maximun epochs
)

########### training loop ###########
try:
    trainer.train()
except (KeyboardInterrupt, SystemExit):
    trainer.save_model()
