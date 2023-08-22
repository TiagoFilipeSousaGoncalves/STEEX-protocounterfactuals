# Imports
import numpy as np
from tqdm import tqdm

# PyTorch Imports
import torch



# Function: Compute accuracy
def compute_accuracy(pred, target):
    same_ids = (pred == target).float().cpu()
    return torch.mean(same_ids,axis=0).numpy()



# Function: Train one epoch (for CelebaDB, CelebaMaskHQDB)
def train_one_epoch_celeba(opt, dataloader_train, model, optimizer, criterion, device):

    print("Number of batches:", len(dataloader_train))
    total_loss = 0
    stat_loss = 0

    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    model.train()

    for batch_idx, batch_data in enumerate(tqdm(dataloader_train)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data['image']

        pred = model(inputs)
        pred_labels = torch.where(pred > 0.5, 1.0, 0.0)
        real_labels = torch.index_select(batch_data['attributes'], 1, torch.tensor(opt.train_attributes_idx).to(device))

        # Compute loss and gradients
        loss = criterion(pred, real_labels)
        acc = compute_accuracy(pred_labels, real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc

        loss.backward()
        optimizer.step()


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean accuracy on the last 50 batches: '+ str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss / len(dataloader_train)
    total_mean_acc = total_acc / len(dataloader_train)
    print('mean loss over training set: %f' % (total_mean_loss))
    print('mean accuracy over training set: ' + str(total_mean_acc))

    return total_mean_loss



# Function: Evaluate one epoch
def evaluate_one_epoch_celeba(opt, dataloader_val, model, criterion, device):

    model.eval()

    total_loss = 0
    stat_loss = 0
    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    print("Number of batches:", len(dataloader_val))

    for batch_idx, batch_data in enumerate(tqdm(dataloader_val)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass

        inputs = batch_data['image']
        with torch.no_grad():
            pred = model(inputs)
            pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

            real_labels = torch.index_select(batch_data['attributes'],1,torch.tensor(opt.train_attributes_idx).to(device))

        # Compute loss and metrics
        loss = criterion(pred,real_labels)
        acc = compute_accuracy(pred_labels,real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean accuracy on the last 50 batches: ' + str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss/len(dataloader_val)
    total_mean_acc = total_acc/len(dataloader_val)

    print('mean loss over validation set: %f'%(total_mean_loss))
    print('mean accuracy over validation set: '+str(total_mean_acc))

    return total_mean_loss



# Function: Train one epoch (for BDDOIADB)
def train_one_epoch_bddoia(opt, dataloader_train, model, optimizer, criterion, device):

    print("Number of batches:", len(dataloader_train))
    total_loss = 0
    stat_loss = 0

    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    model.train()

    for batch_idx, batch_data in enumerate(tqdm(dataloader_train)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['target'] = batch_data['target'].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data['image']

        pred = model(inputs)
        pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

        real_labels = torch.index_select(batch_data['target'],1,torch.tensor(opt.train_attributes_idx).to(device))


        # Compute loss and gradients
        loss = criterion(pred,real_labels)
        acc = compute_accuracy(pred_labels,real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc


        loss.backward()
        optimizer.step()


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean accuracy on the last 50 batches: '+ str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss/len(dataloader_train)
    total_mean_acc = total_acc/len(dataloader_train)
    print('mean loss over training set: %f'%(total_mean_loss))
    print('mean accuracy over training set: ' + str(total_mean_acc))

    return total_mean_loss



# Function: Evaluate one epoch (for BDDOIADB)
def evaluate_one_epoch_bddoia(opt, dataloader_val, model, criterion, device):

    model.eval()

    total_loss = 0
    stat_loss = 0
    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    print("Number of batches:", len(dataloader_val))

    for batch_idx, batch_data in enumerate(tqdm(dataloader_val)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['target'] = batch_data['target'].to(device)

        # Forward pass

        inputs = batch_data['image']
        with torch.no_grad():
            pred = model(inputs)
            pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

            real_labels = torch.index_select(batch_data['target'],1,torch.tensor(opt.train_attributes_idx).to(device))

        # Compute loss and metrics
        loss = criterion(pred,real_labels)
        acc = compute_accuracy(pred_labels,real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean accuracy on the last 50 batches: ' + str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss/len(dataloader_val)
    total_mean_acc = total_acc/len(dataloader_val)

    print('mean loss over validation set: %f'%(total_mean_loss))
    print('mean accuracy over validation set: '+str(total_mean_acc))

    return total_mean_loss
