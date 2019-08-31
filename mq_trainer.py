import os
import torch
import os
import progressbar
from numpy import linalg as LA


class MQTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_generator,
        val_generator,
        exp_index,
        max_epoch,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.train_gen = train_generator
        self.val_gen = val_generator
        self.iter = 0
        self.train_loss = 0.0
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.const_1 = torch.Tensor([1.0]).to(self.device)
        self.exp_folder = "./validation_experiments/mq_" + str(exp_index)
        if not os.path.exists(self.exp_folder):
            os.mkdir(self.exp_folder)

    def train(self):
        for epoch in range(self.max_epoch):
            self.epoch = epoch + 1
            avg_train_loss = self.forward_pass(False)
            print(
                "average train loss:{:.4f} in {} epoch".format(
                    avg_train_loss, self.epoch
                )
            )
            avg_val_loss = self.validate()
            print(
                "average validation loss:{:.4f} in {} epoch".format(
                    avg_val_loss, self.epoch
                )
            )
            self.save_model()

    def validate(self):
        avg_loss = self.forward_pass(True)
        return avg_loss

    def forward_pass(self, b_val):
        if b_val:
            torch.set_grad_enabled(False)
            dataset = self.val_gen
        else:
            torch.set_grad_enabled(True)
            dataset = self.train_gen

        with progressbar.ProgressBar(
            max_value=len(dataset), redirect_stdout=True
        ) as bar:
            total_loss = 0.0
            for y, x in dataset:  # x : mat  y : quaternion
                x, y = (
                    x.to(self.device, dtype=torch.float),
                    y.to(self.device, dtype=torch.float),
                )
                # print("x", x)
                x = x.view(-1, 9)
                # print("x re", x)
                output = self.model(x)
                # output = output.view(-1, 4)
                # print("output ", output.shape)
                # print("y", y.shape)
                # dif = output - y
                l = y.shape[0]
                loss = self.criterion(output, y)
                loss2 = self.criterion(
                    self.const_1.expand(1, l),
                    torch.Tensor([LA.norm(y.to("cpu"), axis=1)]).to(self.device),
                )
                loss += loss2
                total_loss += loss.data
                if b_val:
                    bar.update(self.iter)
                    self.iter += 1
                else:
                    self.train_loss = loss
                    self.optimizer.zero_grad()
                    self.train_loss.backward()
                    self.optimizer.step()
                    bar.update(self.iter)
                    print(
                        "train loss:{:.10f} in {} iteration {} epoch.".format(
                            self.train_loss.data, self.iter, self.epoch
                        )
                    )
                    self.iter += 1

            avg_loss = total_loss / len(dataset)
            self.iter = 0
            return avg_loss

    def save_model(self):
        model_file_name = "mqv1_" + str(self.epoch) + ".pth"
        torch.save(self.model.state_dict(), "./models/" + model_file_name)
        print("Model file: " + model_file_name + " saved.")
