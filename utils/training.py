from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import csv
from datetime import datetime

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
        self.train_metrics = params['metrics']['train'].to(self.device)
        self.valid_metrics = params['metrics']['valid'].to(self.device)
        self.logdir = params['logdir']

        # if params['scaler'] is not None:
        #     self.scaler = GradScaler()
        if self.logdir is None:
            self.writer = SummaryWriter()

        if self.logdir is not None:
            self.logfilename_train = f'{self.logdir}/logs_{datetime.now()}_train.csv'
            self.logfilename_test = f'{self.logdir}/logs_{datetime.now()}_test.csv'
            with open(self.logfilename_train, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['epoch'] +
                    [key for key, _ in sorted(self.train_metrics.items(), key=lambda x: x[0])]
                    )
            with open(self.logfilename_test, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['epoch'] +
                    [key for key, _ in sorted(self.valid_metrics.items(), key=lambda x: x[0])]
                    )

    def optimize(self) -> None:
        for epoch in range(self.n_epochs):
            self.training(
                self.train_loader, self.model,
                self.loss_func, self.optimizer,
                self.train_metrics, self.device)
            self.validation(
                self.test_loader, self.model,
                self.valid_metrics, self.device)
            total_train_metrics = self.train_metrics.compute()
            total_valid_metrics = self.valid_metrics.compute()
            print(f"Training acc for epoch {epoch}: {total_train_metrics}")
            print(f"Validation acc for epoch {epoch}: {total_valid_metrics}")

            if self.logdir is None:
                for metric, value in total_train_metrics.items():
                    self.writer.add_scalar(f'Train/{metric}', value, epoch)
                for metric, value in total_valid_metrics.items():
                    self.writer.add_scalar(f'Test/{metric}', value, epoch)
            
            if self.logdir is not None:
                with open(self.logfilename_train, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                    [str(epoch)] +
                    [value.item() for _, value in sorted(total_train_metrics.items(), key=lambda x: x[0])]
                    )
                with open(self.logfilename_test, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                    [epoch] +
                    [value.item() for _, value in sorted(total_valid_metrics.items(), key=lambda x: x[0])]
                    )

            self.train_metrics.reset()
            self.valid_metrics.reset()


def faster_training_loop(train_loader, model, loss_func, optimizer, metrics, scaler, device: str) -> None:
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
    for _, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        with autocast():
            prediction = model(data)
            loss = loss_func(prediction, labels)
        
        _, pred_labels = prediction.max(dim=1)
        metrics.update(pred_labels, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()


def basic_training_loop(train_loader, model, loss_func, optimizer, metrics, device):
    for _, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        with autocast():
            prediction = model(data)
            loss = loss_func(prediction, labels)
        
        _, pred_labels = prediction.max(dim=1)
        metrics.update(pred_labels, labels)

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
        prediction = model(data)
        loss = criterion(prediction, labels)
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