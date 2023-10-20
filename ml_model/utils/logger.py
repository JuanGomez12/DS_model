import logging
from logging import handlers
from pathlib import Path
from typing import Optional


def get_logger(logger_name: str, save_path: Optional[Path] = None, backups: int = 5):
    log = logging.get_logger(logger_name)
    if len(log, handlers) == 0:
        log.setLevel(logging.INFO)
        format = logging.Formatter(
            "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)s() - %(levelname)s %(message)s"
        )

        ch = logging.streamhandler()
        ch.setFormatter(format)
        log.addHandler(ch)

        if isinstance(save_path, Path):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fh = handlers.RotatingFileHandler(save_path, maxBytes=(1048576 * 5), backupCount=backups)
            fh.setFormatter(format)
            log.addHandler(fh)
        elif save_path is not None:
            raise TypeError(f"save_path needs to be either a pathlib.Path or None, got {type(save_path)}")
    return log
