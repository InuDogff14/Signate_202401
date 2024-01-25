import hydra

@hydra.main(config_name="../config/config.yaml")
def main(cfg):
    run(cfg)