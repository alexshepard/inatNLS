import os
import yaml


class Config:
    @staticmethod
    def load_config():
        env = os.getenv("FLASK_ENV", "development")

        with open("config.yml", "r") as config_file:
            all_configs = yaml.safe_load(config_file)
            config = all_configs["default"]
            config.update(all_configs.get(env, {}))

        return config
