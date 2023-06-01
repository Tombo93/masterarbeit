class OptimizationLoop:
    def __init__(self, params) -> None:
        self.n_epochs = params['n_epochs']
        self.training = params['train_loop']
        self.validation = params['validation_loop']
        
        self.model = params['model']
        self.train_loader = params['train_loader']
        self.test_loader = params['test_loader']
        self.loss_func = params['loss']
        self.optimizer = params['optim']
        self.device = params['device']

    def optimize(self) -> None:
        for _ in range(self.n_epochs):
            self.training(
                self.train_loader, self.model,
                self.loss_func, self.optimizer, self.device)
            self.validation(
                self.train_loader, self.model,
                self.loss_func, self.device)


def basic_training_loop(
        train_loader,
        model,
        loss_func,
        optimizer,
        device: str
        ) -> None:
    """
        use autocast for all forward passes:
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = criterion(scores, labels)
        use a scaler for stabilizing learning (https://github.com/vahidk/EffectivePyTorch#gradscalar):
        scaler = torch.cuda.amp.GradScaler()

        loss = ...
        optimizer = ...  # an instance torch.optim.Optimizer

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    """
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        scores = model(data)
        loss = loss_func(scores, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


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