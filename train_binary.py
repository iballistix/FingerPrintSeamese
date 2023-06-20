import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from RotationDatasetBinary import data_loader_train, data_loader_test
from model import SiameseNetwork

torch.manual_seed(0)
np.random.seed(0)


class Args():
    def __init__(self):
        self.config = 'configs/train_config.yaml'
        self.model_path = None


args = Args()

with open(args.config, 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        quit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss_contrastive = torch.mean((1-label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive


loss_fn = ContrastiveLoss()


def F1(targets, preds):
    return f1_score(targets.cpu().detach().numpy(), (torch.nn.Sigmoid()(preds) > 0.5).cpu().detach().numpy(),
                    average='samples')


def train_step(model, optimizer, data_loader, epoch, batch_accum, device=device):
    model.train()
    running_loss = 0.
    num_data = 0.

    bs = 0
    data_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, (images_1, images_2, targets) in data_loop:
        if i == 0:
            bs = len(images_1)
        images_1 = images_1.to(device)
        images_2 = images_2.to(device)
        targets = targets.to(device)
        targets = torch.unsqueeze(targets, dim=-1).type(torch.float)
        preds = model(images_1, images_2)
        loss = loss_fn(preds, targets)
        loss_value = loss.item()
        running_loss += loss_value

        loss.backward()

        if ((i + 1) % batch_accum) == 0:
            optimizer.step()
            optimizer.zero_grad()

        if i == 0:
            targets_arr = np.array(targets.cpu().detach().numpy())
            preds_arr = np.array((torch.nn.Sigmoid()(preds) > 0.5).cpu().detach().numpy())
        else:
            targets_arr = np.concatenate([targets_arr, np.array(targets.cpu().detach().numpy())])
            preds_arr = np.concatenate([preds_arr, np.array((torch.nn.Sigmoid()(preds) > 0.5).cpu().detach().numpy())])
        num_data += len(preds)

        # Update progress bar
        loop_description = 'Epoch (train) - {}/{}'.format(epoch, cfg['epochs'])
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss_value)
    epoch_f1 = f1_score(targets_arr, preds_arr, average='weighted')
    epoch_loss = running_loss / len(data_loader)
    total_num_data = len(data_loader) * bs
    print('Train Results (Total {})'.format(total_num_data))
    return epoch_loss, epoch_f1


def val_step(model, data_loader, epoch, device=device):
    model.eval()
    running_loss = 0.
    num_data = 0.

    data_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    data_loop.set_description('Epoch {} - Validation'.format(epoch))
    for i, (images_1, images_2, targets) in data_loop:
        images_1 = images_1.to(device)
        images_2 = images_2.to(device)
        targets = targets.to(device)
        targets = torch.unsqueeze(targets, dim=-1).type(torch.float)
        preds = model(images_1, images_2)

        loss = loss_fn(preds, targets)
        loss_value = loss.item()
        running_loss += loss_value

        if i == 0:
            targets_arr = np.array(targets.cpu().detach().numpy())
            preds_arr = np.array((torch.nn.Sigmoid()(preds) > 0.5).cpu().detach().numpy())
        else:
            targets_arr = np.concatenate([targets_arr, np.array(targets.cpu().detach().numpy())])
            preds_arr = np.concatenate([preds_arr, np.array((torch.nn.Sigmoid()(preds) > 0.5).cpu().detach().numpy())])
        num_data += len(preds)

    val_f1 = f1_score(targets_arr, preds_arr, average='weighted')
    val_loss = running_loss / len(data_loader)
    print('Validation Results (Total {})'.format(len(data_loader)))

    return val_loss, val_f1


def train_model(model, cfg):
    if not os.path.exists(cfg['model_save_root']):
        os.makedirs(cfg['model_save_root'])
    TRAIN_SAVE_PATH = os.path.join(cfg['model_save_root'], cfg['exp_name'] + '_train.pt')
    VAL_SAVE_PATH = os.path.join(cfg['model_save_root'], cfg['exp_name'] + '.pt')

    #params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.net_parameters, cfg['lr'])

    best_loss = 1000
    patience_lvl = 0
    start_epoch = 0
    try:
        start_epoch = cfg['cont_epoch']
        print('Continue training at Epoch: {}'.format(start_epoch))
    except:
        pass

    writer = SummaryWriter(os.path.join('logs', cfg['exp_name']))
    for epoch in range(start_epoch, start_epoch + cfg['epochs']):
        try:
            train_dataloader = data_loader_train(cfg['train_label_path'], cfg['img_size'],
                                                                     cfg['batch_size'])
            test_dataloader = data_loader_test(cfg['test_label_path'], cfg['img_size'], 1)

            train_loss, train_f1 = train_step(model, optimizer,
                                              train_dataloader,
                                              epoch,
                                              cfg['batch_accum'])
            writer.add_scalar('loss/train_loss', train_loss, global_step=epoch)
            writer.add_scalar('acc/train_f1', train_f1, global_step=epoch)

            test_loss, test_f1 = val_step(model, test_dataloader, epoch)

            writer.add_scalar('loss/test_loss', test_loss, global_step=epoch)
            writer.add_scalar('acc/test_f1', test_f1, global_step=epoch)

            print('Epoch {} - Train Loss: {} - Train F1: {} - Test Loss: {} - Test F1: {} -'
                  .format(epoch, train_loss, train_f1, test_loss, test_f1))

            if test_loss > best_loss:
                patience_lvl += 1
                # Early Stopping
                if patience_lvl >= cfg['patience']:
                    print('Early stopping at Epoch {} - Best Loss: {}'.format(epoch, best_loss))
                    break
            else:
                patience_lvl = 0
                best_loss = test_loss
                # Save best model
                torch.save(model.state_dict(), VAL_SAVE_PATH)
            torch.save(model.state_dict(), TRAIN_SAVE_PATH)
        except KeyboardInterrupt:
            print('Training stopped at epoch {}'.format(epoch - 1))
            return VAL_SAVE_PATH
    return VAL_SAVE_PATH


if __name__ == '__main__':
    model = SiameseNetwork(lastLayer=False, pretrained=True)
    if args.model_path is not None:
        print('Loading model from {}'.format(args.model_path))
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    train_model(model, cfg)
