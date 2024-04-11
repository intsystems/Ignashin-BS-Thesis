def trainer(model , path = None, init_weights=init_weights , N_EPOCHS=1 , CLIP=1  ,):
    model.apply(init_weights)
    print(count_parameters(model))

    PAD_IDX = TRG.vocab.stoi['<pad>']
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    train_history = []
    valid_history = []
    train_epoch_history = []
    valid_epoch_history = []

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss_epoch , train_losses = train(model, train_iterator, optimizer, criterion, CLIP, train_history, valid_history)
        valid_loss_epoch , val_losses = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        train_epoch_history.append(train_loss_epoch)
        valid_epoch_history.append(valid_loss_epoch)

        train_history += train_losses
        valid_history += val_losses
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss_epoch:.3f} | Train PPL: {math.exp(train_loss_epoch):7.3f}')
        print(f'\t Val. Loss: {valid_loss_epoch:.3f} |  Val. PPL: {math.exp(valid_loss_epoch):7.3f}')
    return train_history, valid_history , train_epoch_history , valid_epoch_history