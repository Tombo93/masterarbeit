from dataclasses import dataclass


@dataclass
class AverageMeter:
    val: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
