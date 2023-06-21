import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from RotationDatasetBinary import data_loader_train, data_loader_test
from model import SiameseNetwork

torch.manual_seed(0)
np.random.seed(0)


class Args():
    def __init__(self):
        self.config = 'configs/train_config.yaml'
        self.model_path = 'saved_models/exp1.pt'


args = Args()

with open(args.config, 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        quit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive

loss_fn = ContrastiveLoss()
#loss_fn = torch.nn.BCELoss()
dist = torch.nn.PairwiseDistance()

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
        output1, output2 = model(images_1, images_2)

        loss = loss_fn(output1, output2, targets)

        #loss = loss_fn(scores, targets)
        loss_value = loss.item()
        running_loss += loss_value

        loss.backward()

        if ((i + 1) % batch_accum) == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update progress bar
        loop_description = 'Epoch (train) - {}/{}'.format(epoch, cfg['epochs'])
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss_value)
    epoch_loss = running_loss / len(data_loader)
    total_num_data = len(data_loader) * bs
    print('Train Results (Total {})'.format(total_num_data))
    return epoch_loss


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
        output1, output2 = model(images_1, images_2)

        loss = loss_fn(output1, output2, targets)
        loss_value = loss.item()
        running_loss += loss_value



    val_loss = running_loss / len(data_loader)
    print('Validation Results (Total {})'.format(len(data_loader)))

    return val_loss


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

            train_loss = train_step(model, optimizer,
                                              train_dataloader,
                                              epoch,
                                              cfg['batch_accum'])
            writer.add_scalar('loss/train_loss', train_loss, global_step=epoch)


            test_loss = val_step(model, test_dataloader, epoch)

            writer.add_scalar('loss/test_loss', test_loss, global_step=epoch)


            print('Epoch {} - Train Loss: {} - Test Loss: {}'
                  .format(epoch, train_loss, test_loss))

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
