import logging
from pathlib import Path
from pprint import pformat
from typing import Optional

import yaml


def load_config_file(config_path: Path, logger: Optional[logging.Logger] = logging.getLogger(__name__)) -> dict:
    """Returns the config yaml file as a dictionary.

    Args:
        config_path (Path): Path to the ocnfiguration file.
        logger (Optional[logging.Logger], optional): Logger to use to log information. If None, it won't log.
            Defaults to logging.getLogger(__name__).

    Returns:
        dict: Dictionary contianing the parameters found in the input yaml file.
    """
    if logger is not None:
        logger.info("Loading config")
    with open(config_path) as f:
        config = yaml.full_load(f)
    if logger is not None:
        logger.info("Config loaded")
        logger.info(f"{pformat(config)}")
    return config
