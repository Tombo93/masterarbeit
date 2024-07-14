import hydra
from hydra.core.config_store import ConfigStore

from models import isic_main, isic_backdoor_main, isic_main_lightning
from data import make_isic, make_isic_metadata
from visualization import plot_confusion_matrix, plot_isic
from config import Config

CS = ConfigStore.instance()
CS.store(name="isic_config", node=Config)

DEBUG = False
SAVE_BASE_MODEL = False


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    # print("Generating appropriate metadata..")
    # make_isic_metadata.main(cfg.preprocessing)

    # print("Preprocess the data..")
    # make_isic.main(cfg)

    # print("Setting up experiment..")
    # isic_main.main(cfg, save_model=SAVE_BASE_MODEL, debug=DEBUG)
    # print("Training on poisoned data..")
    isic_backdoor_main.main(cfg, debug=DEBUG)

    # print("Plotting data..")
    # plot_confusion_matrix.main(cfg)
    # plot_isic.main(plot_cfg=cfg.plotting)


if __name__ == "__main__":
    main()
