import numpy as np
import torch
from tqdm import tqdm, trange


log = tqdm.write


def get_device(model):
    return next(model.parameters()).device.type


def train(model, data_loader, log_level=3):
    """
    This function trains a `model` given a `data_loader` by `optimizer`
    """
    model.train()

    train_loss = {}

    for batch_data in tqdm(data_loader, total=len(data_loader),
                           ascii=True, disable=log_level < 1):
        data = {"X": torch.from_numpy(batch_data[0].values),
                "keys": batch_data[0].columns.values}
        # Batch size of length 1 is not allowed due to the batch normalization used in the models
        if data["X"].shape[0] == 1:
            continue

        if get_device(model) == "cuda":
            data["X"] = data["X"].cuda()

        with torch.autograd.detect_anomaly():
            loss = model.train_on(data)

        for loss_type in loss:
            if loss_type not in train_loss:
                train_loss[loss_type] = 0
            train_loss[loss_type] += loss[loss_type]

    for loss_type in train_loss:
        train_loss[loss_type] /= len(data_loader.dataset)

    if log_level >= 2:
        log('====> Train | model loss: {}'.format(str(train_loss)))
    return train_loss


def test(model, data_loader, log_level=3):
    """
    This function tests a `model` on a `data_loader`
    """
    model.eval()

    test_loss = {}

    with torch.no_grad():
        for batch_data in data_loader:
            data = {"X": torch.from_numpy(batch_data[0].values),
                    "keys": batch_data[0].columns.values}

            if get_device(model) == "cuda":
                data["X"] = data["X"].cuda()

            output = model.get_output(data)
            loss = model.loss(data, output)
            for loss_type in loss:
                if loss_type not in test_loss:
                    test_loss[loss_type] = 0
                test_loss[loss_type] += loss[loss_type].item()

    for loss_type in test_loss:
        test_loss[loss_type] /= len(data_loader.dataset)

    if log_level >= 2:
        log('====> Test  | model loss: {:.4f}'.format(test_loss["loss"]))
    return test_loss


def run(model, n_epoch, train_loader, test_loader, log_level=3):
    """
    This function will optimize `model` for `n_epoch` epochs
    on `train_loader` dataset and validate it on `test_loader`.
    """

    for epoch in trange(1, n_epoch + 1, unit="epoch", ascii=True, disable=log_level < 1):
        if log_level >= 1:
            log("Epoch: %d" % epoch)
        model.set_epoch(epoch)
        train_loss = train(model, train_loader, log_level=log_level)
        test_loss = test(model, test_loader, log_level=log_level)
        signal = model.finalize_train(train_loss, test_loss)

        if signal is not None:
            if signal == "stop":
                if log_level >= 1:
                    log("EarlyStopping: Stop training")
                break
            else:
                if log_level >= 1:
                    log(signal)


def get_output(model, data_loader):
    model.eval()

    combined_outputs = dict()

    with torch.no_grad():
        for batch_data in data_loader:
            data = {"X": torch.from_numpy(batch_data[0].values),
                    "keys": batch_data[0].columns.values}

            if get_device(model) == "cuda":
                data["X"] = data["X"].cuda()

            outputs = model.get_output(data)
            for key in outputs:
                output = outputs[key].detach().cpu().numpy()
                if key not in combined_outputs:
                    combined_outputs[key] = output
                else:
                    combined_outputs[key] = np.vstack([combined_outputs[key], output])

    return combined_outputs
