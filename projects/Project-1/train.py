from tqdm import tqdm
import torch
from utils.utils import save_checkpoint, load_checkpoint

def train_batch(imgs, ages, genders, model, optimizer, criterion_mse, criterion_bce, scaler = None):
    loss_mse = 0.0
    loss_bce = 0.0
    if scaler:
        with torch.cuda.amp.autocast():
            pred_age, pred_genders = model(imgs)
            loss_mse = criterion_mse(pred_age.squeeze(), ages)
            loss_bce = criterion_bce(pred_genders.squeeze(), genders)
            total_loss = loss_mse + loss_bce
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred_age, pred_genders = model(imgs)
        loss_mse = criterion_mse(pred_age.squeeze(), ages)
        loss_bce = criterion_bce(pred_genders.squeeze(), genders)
        total_loss = loss_mse + loss_bce
        total_loss.backward()
        optimizer.step()
    return loss_mse.item(), loss_bce.item(), total_loss.item()


def validate_batch(imgs, ages, genders, model, criterion_mse, criterion_bce):
    loss_mse = 0.0
    loss_bce = 0.0
    pred_age, pred_genders = model(imgs)
    loss_mse = criterion_mse(pred_age.squeeze(), ages)
    loss_bce = criterion_bce(pred_genders.squeeze(), genders)
    total_loss = loss_mse + loss_bce
    return loss_mse.item(), loss_bce.item(), total_loss.item()

def train(trainloader, validloader, model, criterion_mse, criterion_bce, optimizer, epochs, best_loss = 999999, scheduler = None):
    '''
    Train the network
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_mse_losses, train_bce_losses = [], []
    valid_mse_losses, valid_bce_losses = [], []
    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    model = model.to(device)
    for epoch in range(epochs+1):
        batch_mse_loss = 0.0
        batch_bce_loss = 0.0
        model.train()
        pbar = tqdm(trainloader, total = len(trainloader), leave = False)
        for imgs, ages, genders in pbar:
            imgs = imgs.to(device, non_blocking = True)
            ages = ages.to(device, non_blocking = True).float()
            genders = genders.to(device, non_blocking = True).float()
            current_mse, current_bce, _ = train_batch(imgs, ages, genders, model, optimizer, criterion_mse, criterion_bce, scaler)
            batch_mse_loss+=current_mse
            batch_bce_loss+=current_bce
            pbar.set_postfix(bce_loss = current_bce, mse_loss = current_mse)
        batch_mse_loss /= len(trainloader)
        batch_bce_loss /= len(trainloader)
        train_mse_losses.append(batch_mse_loss)
        train_bce_losses.append(batch_bce_loss)

        valid_mse_loss = 0.0
        valid_bce_loss = 0.0
        valid_total_loss = 0.0
        pbar = tqdm(validloader, total = len(validloader), leave = False)
        for imgs, ages, genders in pbar:
            imgs = imgs.to(device, non_blocking = True)
            ages = ages.to(device, non_blocking = True).float()
            genders = genders.to(device, non_blocking = True).float()
            current_mse, current_bce, current_loss= validate_batch(imgs, ages, genders, model, criterion_mse, criterion_bce)
            valid_mse_loss+=current_mse
            valid_bce_loss+=current_bce
            valid_total_loss += current_loss
            pbar.set_postfix(bce_valid_loss = current_bce, mse_valid_loss = current_mse, total_valid_loss = current_loss)
        
        valid_mse_loss /= len(validloader)
        valid_bce_loss /= len(validloader)
        valid_total_loss /= len(validloader)
        valid_mse_losses.append(valid_mse_loss)
        valid_bce_losses.append(valid_bce_loss)

        # Checkpoint Saving
        if valid_total_loss <= best_loss:
            best_loss = valid_total_loss
            save_checkpoint(model, optimizer, 'saved_models/multi_task.pt')

        if scheduler:
            scheduler.step()

        print(f'''[{epoch:2}/{epochs}]:
Train BCE Loss: {batch_bce_loss:.4f}, Train MSE Loss: {batch_mse_loss:.4f}
Valid BCE Loss: {valid_bce_loss:.4f}, Valid MSE Loss: {valid_mse_loss:.4f}''')

    model = load_checkpoint(model, file_path='saved_models/multi_task.pt')
    return model, train_bce_losses, train_mse_losses, valid_bce_losses, valid_mse_losses, best_loss
        