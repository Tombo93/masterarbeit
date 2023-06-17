import csv
from datetime import datetime

from utils.training import Training
from utils.evaluation import Validation

from torch.utils.tensorboard import SummaryWriter


class OptimizationLoop:
    def __init__(self,
                 params,
                 training: Training,
                 validation: Validation) -> None:
        
        self.n_epochs = params['n_epochs']
        self.training = training
        self.validation = validation
        
        self.model = params['model']
        self.train_loader = params['train_loader']
        self.test_loader = params['test_loader']
        self.loss_func = params['loss']
        self.optimizer = params['optim']
        self.device = params['device']
        self.train_metrics = params['metrics']['train'].to(self.device)
        self.valid_metrics = params['metrics']['valid'].to(self.device)
        self.logdir = params['logdir']

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
            self.training.run(
                self.train_loader, self.model,
                self.loss_func, self.optimizer,
                self.train_metrics, self.device)
            self.validation.run(
                self.test_loader, self.model,
                self.valid_metrics, self.device)
            total_train_metrics = self.train_metrics.compute()
            total_valid_metrics = self.valid_metrics.compute()
            print(f"Training metrics for epoch {epoch}: {total_train_metrics}")
            print(f"Validation metrics for epoch {epoch}: {total_valid_metrics}")

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
