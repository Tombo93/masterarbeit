import os

import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from config import Config

import isic_main
from data import make_isic, make_isic_metadata

CS = ConfigStore.instance()
CS.store(name="isic_config", node=Config)
OmegaConf.register_new_resolver("join", lambda x, y: os.path.join(x, y))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))

    # TODO:
    # 1. orchestrate metadata preprocessing
    # make_isic_metadata.main(cfg=cfg.preprocessing)
    # 2. orchestrate data preprocessing
    make_isic.main(cfg=cfg)
    # 3. orchestrate training
    # isic_main.main(cfg)
    # cls = cfg.data.classes
    # 4. orchestrate reporting
    # print(cfg.reports.test_report)


# Read data spec
# create metadata
# create image+label data from metadata
# Read training spec
# start training
# log results
# plot results

if __name__ == "__main__":
    main()
