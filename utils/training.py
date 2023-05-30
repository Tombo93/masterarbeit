def basic_training_loop(
        epochs: int,
        train_loader,
        model,
        criterion,
        optimizer,
        device: str
        ) -> None:
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):

            data = data.to(device)
            labels = labels.to(device)

            # predict classes and calculate loss
            scores = model(data)
            loss = criterion(scores, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            # update model parameters
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


def single_batch_test(
        epochs: int,
        train_loader,
        model,
        criterion,
        optimizer,
        device: str
        ) -> float:
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        data, labels = next(iter(train_loader))

        data = data.to(device)
        labels = labels.to(device)
        # predict classes and calculate loss
        scores = model(data)
        loss = criterion(scores, labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        # update model parameters
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if epoch % 10 == 0:
            print(f'[{epoch + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        return running_loss / 10