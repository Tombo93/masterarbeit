import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_line(file_path: str, metric: str) -> None:
    data = pd.read_csv(file_path)

    plt.figure(dpi=300)
    plt.plot(np.arange(data.shape[0]), data[metric])
    plt.xlabel('n epochs')
    plt.ylabel(f'{metric}')
    plt.ylim(0,1)
    plt.grid(False)
    plt.savefig('XYZ.png')


def plot_multi_line(file_paths, metric, plot_loss=False):
    plt.figure(dpi=300)

    for file_path in file_paths:
        data = pd.read_csv(file_path)
        plt.plot(np.arange(data.shape[0]), data[metric].multiply(100), label=f"{file_path.split('-model-')[-1]}")
    
    plt.xlabel('n epochs')
    plt.ylabel(f'{metric} (%)')
    if not plot_loss:
        plt.ylim(0,100)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{metric}.png")


def main():
    csv_multi_paths = [
        "/home/bay1989/masterarbeit/experiments/Multi-ValMetrics20231030_ISIC_ccr_corrected_two_labels-model-resnet50-finetuning-batchsize-32-lr-0.001.csv",
        "/home/bay1989/masterarbeit/experiments/Multi-ValMetrics20231030_ISIC_ccr_corrected_two_labels-model-resnet50-finetuning-batchsize-32-lr-0.0001.csv",
        "/home/bay1989/masterarbeit/experiments/Multi-ValMetrics20231030_ISIC_ccr_corrected_two_labels-model-resnet50-finetuning-batchsize-64-lr-0.001.csv",
        "/home/bay1989/masterarbeit/experiments/Multi-ValMetrics20231030_ISIC_ccr_corrected_two_labels-model-resnet50-finetuning-batchsize-64-lr-0.0001.csv",
        "/home/bay1989/masterarbeit/experiments/Multi-ValMetrics20231030_ISIC_ccr_corrected_two_labels-model-BatchNormCNN-batchsize-32-lr-0.001.csv"
    ]
    metrics = ['BinaryAUROC', 'BinaryRecall', 'BinaryAccuracy', 'BinaryPrecision', 'BinaryFBetaScore']
    for metric in metrics:
        plot_multi_line(csv_multi_paths, metric)
    plot_multi_line(csv_multi_paths, 'Loss', auto_limits=True)


if __name__ == "__main__":
    main()