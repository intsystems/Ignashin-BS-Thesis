
import torch 
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import numpy as np




def train_on_batch(model, x_batch, y_batch, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()
    
    print(x_batch.shape , y_batch.shape)
    print(x_batch , y_batch)
    x_batch =x_batch.permute(1,0)
    y_batch =y_batch.permute(1,0)
    output = model(x_batch.to(model.device), y_batch.to(model.device))
    

    print(output.transpose(1,2)[0] , y_batch[0] )
    loss = loss_function(output.transpose(1,2), 
                         y_batch.to(model.device))
    loss.backward()

    optimizer.step()
    return loss.cpu().item()

def train_epoch(train_generator, model, loss_function, optimizer, callback = None):
    epoch_loss = 0
    total = 0
    for it, (batch_of_x, batch_of_y) in enumerate(train_generator):
        batch_loss = train_on_batch(model, batch_of_x, batch_of_y, optimizer, loss_function)
        
        if callback is not None:
            with torch.no_grad():
                callback(model, batch_loss)
            
        epoch_loss += batch_loss*len(batch_of_x)
        total += len(batch_of_x)
    
    return epoch_loss/total


def trainer(count_of_epoch, 
            batch_size, 
            dataloader,
            dataset , 
            model, 
            loss_function,
            optimizer,
            lr = 0.001,
            callback = None):

    optima = optimizer(model.parameters(), lr=lr)
    
    # iterations = tqdm(range(count_of_epoch), desc='epoch')
    # iterations.set_postfix({'train epoch loss': np.nan})
    iterations = count_of_epoch
    for it in range(iterations):
        # batch_generator = tqdm(
        #     dataloader, 
        #     leave=False, total=len(dataset)//batch_size+(len(dataset)%batch_size>0))
        batch_generator = dataloader
        epoch_loss = train_epoch(train_generator=batch_generator, 
                    model=model, 
                    loss_function=loss_function, 
                    optimizer=optima, 
                    callback=callback)
        print('train epoch loss: ', epoch_loss)
        # iterations.set_postfix({'train epoch loss': epoch_loss})