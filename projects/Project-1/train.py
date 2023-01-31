from tqdm import tqdm
import torch
from utils.utils import save_checkpoint, load_checkpoint

def train_batch(imgs, ages, genders, model, optimizer, criterion_mae, criterion_bce, scaler = None):
    loss_mae = 0.0
    loss_bce = 0.0
    if scaler:
        with torch.cuda.amp.autocast():
            pred_age, pred_genders = model(imgs)
            loss_mae = criterion_mae(pred_age.squeeze(), ages)
            loss_bce = criterion_bce(pred_genders.squeeze(), genders)
            total_loss = loss_mae + loss_bce
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred_age, pred_genders = model(imgs)
        loss_mae = criterion_mae(pred_age.squeeze(), ages)
        loss_bce = criterion_bce(pred_genders.squeeze(), genders)
        total_loss = loss_mae + loss_bce
        total_loss.backward()
        optimizer.step()
    return loss_mae.item(), loss_bce.item(), total_loss.item()


def validate_batch(imgs, ages, genders, model, criterion_mae, criterion_bce):
    loss_mae = 0.0
    loss_bce = 0.0
    pred_age, pred_genders = model(imgs)
    loss_mae = criterion_mae(pred_age.squeeze(), ages)
    loss_bce = criterion_bce(pred_genders.squeeze(), genders)
    total_loss = loss_mae + loss_bce
    return loss_mae.item(), loss_bce.item(), total_loss.item()

def train(trainloader, validloader, model, criterion_mae, criterion_bce, optimizer, epochs, best_loss = 999999, scheduler = None):
    '''
    Train the network
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_mae_losses, train_bce_losses = [], []
    valid_mae_losses, valid_bce_losses = [], []
    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    model = model.to(device)
    for epoch in range(epochs+1):
        batch_mae_loss = 0.0
        batch_bce_loss = 0.0
        model.train()
        pbar = tqdm(trainloader, total = len(trainloader), leave = False)
        for imgs, ages, genders in pbar:
            imgs = imgs.to(device, non_blocking = True)
            ages = ages.to(device, non_blocking = True).float()
            genders = genders.to(device, non_blocking = True).float()
            current_mae, current_bce, _ = train_batch(imgs, ages, genders, model, optimizer, criterion_mae, criterion_bce, scaler)
            batch_mae_loss+=current_mae
            batch_bce_loss+=current_bce
            pbar.set_postfix(bce_loss = current_bce, mae_loss = current_mae)
        batch_mae_loss /= len(trainloader)
        batch_bce_loss /= len(trainloader)
        train_mae_losses.append(batch_mae_loss)
        train_bce_losses.append(batch_bce_loss)

        valid_mae_loss = 0.0
        valid_bce_loss = 0.0
        valid_total_loss = 0.0
        pbar = tqdm(validloader, total = len(validloader), leave = False)
        for imgs, ages, genders in pbar:
            imgs = imgs.to(device, non_blocking = True)
            ages = ages.to(device, non_blocking = True).float()
            genders = genders.to(device, non_blocking = True).float()
            current_mae, current_bce, current_loss= validate_batch(imgs, ages, genders, model, criterion_mae, criterion_bce)
            valid_mae_loss+=current_mae
            valid_bce_loss+=current_bce
            valid_total_loss += current_loss
            pbar.set_postfix(bce_valid_loss = current_bce, mae_valid_loss = current_mae, total_valid_loss = current_loss)
        
        valid_mae_loss /= len(validloader)
        valid_bce_loss /= len(validloader)
        valid_total_loss /= len(validloader)
        valid_mae_losses.append(valid_mae_loss)
        valid_bce_losses.append(valid_bce_loss)

        # Checkpoint Saving
        if valid_total_loss <= best_loss:
            best_loss = valid_total_loss
            save_checkpoint(model, optimizer, 'saved_models/multi_task.pt')

        if scheduler:
            scheduler.step()

        print(f'''[{epoch:2}/{epochs}]:
Train BCE Loss: {batch_bce_loss:.4f}, Train mae Loss: {batch_mae_loss:.4f}
Valid BCE Loss: {valid_bce_loss:.4f}, Valid mae Loss: {valid_mae_loss:.4f}''')

    model = load_checkpoint(model, file_path='saved_models/multi_task.pt')
    return model, train_bce_losses, train_mae_losses, valid_bce_losses, valid_mae_losses, best_loss
        