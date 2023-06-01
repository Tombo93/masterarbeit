import torch

def basic_validation(test_loader, model, loss_func, device):
    num_correct, num_samples = 0, 0
    test_loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)

            pred = model(x)
            test_loss += loss_func(pred, y).item()
            _, predictions = pred.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            # if test_loss != 0:
            #     break
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()
