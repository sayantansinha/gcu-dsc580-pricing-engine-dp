import toml


def load_config_from_file(config_file: str) -> dict:
    """
    Load TOML configuration file
    :param config_file: toml config file path relevant to the root directory of the project
    :return: dictionary representation of the config file
    """
    return toml.load(config_file)
