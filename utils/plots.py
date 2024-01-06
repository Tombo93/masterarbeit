import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_line(file_path: str, metric: str) -> None:
    data = pd.read_csv(file_path)

    plt.figure(dpi=300)
    plt.plot(np.arange(data.shape[0]), data[metric])
    plt.xlabel('n epochs')
    plt.ylabel(f'{metric}')
    plt.grid(False)
    plt.savefig('XYZ.png')


def main():
    csv_path = "/home/bay1989/masterarbeit/experiments/Multi-TrainMetrics20231030_ISIC_ccr_corrected_two_labels-model-BatchNormCNN-batchsize-32-lr-0.001.csv"
    metrics = ['BinaryAUROC', 'BinaryRecall', 'BinaryAccuracy', 'BinaryPrecision', 'BinaryFBetaScore', 'Loss']
    plot_line(csv_path, 'BinaryAUROC')


if __name__ == "__main__":
    main()