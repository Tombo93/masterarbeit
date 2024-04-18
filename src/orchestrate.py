import click
import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from config import Config

import isic_main, isic_backdoor_main
from data import make_isic, make_isic_metadata
from visualization import plot_confusion_matrix, plot_isic

CS = ConfigStore.instance()
CS.store(name="isic_config", node=Config)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))

    # 1. orchestrate metadata preprocessing
    print("Generating appropriate metadata..")
    make_isic_metadata.main(cfg=cfg.preprocessing)
    # 2. orchestrate data preprocessing
    print("Preprocess the data..")
    make_isic.main(cfg=cfg)
    # 3. orchestrate training
    print("Setting up experiment..")
    isic_main.main(cfg)
    isic_backdoor_main.main(cfg)
    # 4. orchestrate reporting
    print("Plotting data..")
    plot_confusion_matrix.main()
    plot_isic.main()


if __name__ == "__main__":
    main()
