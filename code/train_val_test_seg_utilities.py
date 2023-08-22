# Imports
import numpy as np
from tqdm import tqdm

# PyTorch Imports
import torch



# Function: Compute IoU
def compute_iou(pred, target, opt):

    n_classes = opt.n_classes
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds * target_inds).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)



# Function: Compute accuracy
def compute_accuracy(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    same_ids = pred == target
    
    return same_ids.long().sum().data.cpu().item()/float(torch.numel(pred))



# Function: Training loop
def train_one_epoch(opt, dataloader_train, model, device, optimizer, criterion):

    print("Number of batches:", len(dataloader_train))
    total_loss = 0
    stat_loss = 0
    total_iou = 0
    stat_iou = 0
    total_acc = 0
    stat_acc = 0
    model.train()

    for batch_idx, batch_data in enumerate(tqdm(dataloader_train)):
        batch_data['image'] = batch_data['image'].to(device)
        # print(batch_data['label'].shape)
        batch_data['label'] = batch_data['label'].squeeze(1).long().to(device)
        # print(batch_data['label'].shape)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data['image']

        pred = model(inputs)['out']
        pred_labels = pred.argmax(1)


        # Compute loss and gradients
        loss = criterion(pred,batch_data['label'])
        loss.backward()
        optimizer.step()

        iou = np.nanmean(compute_iou(pred_labels, batch_data['label'], opt))
        acc = compute_accuracy(pred_labels, batch_data['label'])

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_iou += iou.item()
        total_iou += iou.item()
        stat_acc += acc
        total_acc += acc


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean IoU on the last 50 batches: %f'%(stat_iou/batch_interval))
            print('mean pixel accuracy on the last 50 batches: %f'%(stat_acc/batch_interval))
            stat_loss = 0
            stat_iou = 0
            stat_acc = 0

    total_mean_loss = total_loss/len(dataloader_train)
    total_mean_iou = total_iou/len(dataloader_train)
    total_mean_acc = total_acc/len(dataloader_train)
    print('mean loss over training set: %f'%(total_mean_loss))
    print('mean IoU over training set: %f'%(total_mean_iou))
    print('mean pixel accuracy over training set: %f'%(total_mean_acc))

    return total_mean_loss



# Function: Evaluation loop
def evaluate_one_epoch(opt, dataloader_val, model, device, criterion):

    model.eval()

    total_loss = 0
    stat_loss = 0
    total_iou = 0
    stat_iou = 0
    total_acc = 0
    stat_acc = 0

    print("Number of batches:", len(dataloader_val))
    for batch_idx, batch_data in enumerate(tqdm(dataloader_val)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['label'] = batch_data['label'].squeeze(1).long().to(device)

        # Forward pass
        inputs = batch_data['image']
        with torch.no_grad():
            pred = model(inputs)['out']
            pred_labels = pred.argmax(1)

        # Compute loss and metrics
        loss = criterion(pred,batch_data['label'])
        iou = np.nanmean(compute_iou(pred_labels, batch_data['label'], opt))
        acc = compute_accuracy(pred_labels, batch_data['label'])


        stat_loss += loss.item()
        total_loss += loss.item()
        stat_iou += iou.item()
        total_iou += iou.item()
        stat_acc += acc
        total_acc += acc

        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean IoU on the last 50 batches: %f'%(stat_iou/batch_interval))
            print('mean pixel accuracy on the last 50 batches: %f'%(stat_acc/batch_interval))
            stat_loss = 0
            stat_iou = 0
            stat_acc = 0

    total_mean_loss = total_loss/len(dataloader_val)
    total_mean_iou = total_iou/len(dataloader_val)
    total_mean_acc = total_acc/len(dataloader_val)
    print('mean loss over validation set: %f' % (total_mean_loss))
    print('mean IoU over validation set: %f' % (total_mean_iou))
    print('mean pixel accuracy over training set: %f' % (total_mean_acc))
    
    return total_mean_loss
