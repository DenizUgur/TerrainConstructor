from dataset import TerrainDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os

from utils.inpainting_utils import *
from models.skip import skip

hyperparameter_defaults = dict(
    batch_size=4,
    epochs=10,
    learning_rate=0.001,
    train_samples=1000,
    filter_size_up=3,
    filter_size_down=5,
    input_channels=128,
)
wandb.init(config=hyperparameter_defaults, project="trc-1")
config = wandb.config


def train():
    # pylint: disable=no-member
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TDS_T = TerrainDataset(
        "data/MDRS/data/*.tif",
        dataset_type="train",
        randomize=False,
        limit_samples=config.train_samples,
    )
    trainLoader = DataLoader(dataset=TDS_T, batch_size=config.batch_size, num_workers=0)

    TDS_V = TerrainDataset(
        "data/MDRS/data/*.tif",
        dataset_type="validation",
        randomize=False,
        limit_samples=config.train_samples // 2,
    )
    valLoader = DataLoader(dataset=TDS_V, batch_size=config.batch_size, num_workers=0)

    IC = config.input_channels
    net = skip(
        1,
        1,
        num_channels_down=[IC // 8, IC // 4, IC // 2, IC, IC, IC],
        num_channels_up=[IC // 8, IC // 4, IC // 2, IC, IC, IC],
        num_channels_skip=[0, 0, 0, 0, 0, 0],
        filter_size_up=config.filter_size_up,
        filter_size_down=config.filter_size_down,
        filter_skip_size=1,
        upsample_mode="nearest",  # downsample_mode='avg',
        need1x1_up=False,
        need_sigmoid=True,
        need_bias=True,
        pad="reflection",
        act_fun="LeakyReLU",
    ).to(device)
    wandb.watch(net)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    # * Information
    params = sum(np.prod(list(p.size())) for p in net.parameters())
    print("# of parameters = {:,}".format(params))

    # * Training Loop parameters
    bar = False

    # reset = True
    # model_name = "full"
    # if os.path.exists("tds-1/{}.pt".format(model_name)):
    #     if reset:
    #         os.remove("tds-1/{}.pt".format(model_name))
    #     else:
    #         checkpoint = torch.load("tds-1/{}.pt".format(model_name))
    #         net.load_state_dict(checkpoint["model_state_dict"])
    #         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoch in tqdm(range(config.epochs), position=0, disable=not bar):
        ###################
        # train the model #
        ###################
        net.train()
        for i, ((data, _), target) in tqdm(
            enumerate(trainLoader), total=len(trainLoader), position=1, disable=not bar
        ):
            # send iteration data to GPU
            data = data.cuda()
            target = target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_loss_x": epoch * len(trainLoader) + i,
                }
            )

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        for i, ((data, _), target) in tqdm(
            enumerate(valLoader), total=len(valLoader), position=2, disable=not bar
        ):
            # send iteration data to GPU
            data = data.cuda()
            target = target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            wandb.log(
                {
                    "validation_loss": loss.item(),
                    "val_loss_x": epoch * len(valLoader) + i,
                }
            )

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        wandb.log(
            {
                "avg_train_loss": train_loss,
                "avg_valid_loss": valid_loss,
                "avg_loss_x": epoch,
            }
        )

        epoch_len = len(str(config.epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{config.epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"valid_loss: {valid_loss:.5f}"
        )

        if not bar:
            print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []


if __name__ == "__main__":
    train()