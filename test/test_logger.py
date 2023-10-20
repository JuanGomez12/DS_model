import logging

import pytest

from ml_model.utils.logger import get_logger


class TestLogger:
    def test_logger(self, tmp_path):
        log_file_path = tmp_path / "test_log.log"
        logger = get_logger("test_case_1", log_file_path)
        assert logger.handlers[1].baseFilename == str(log_file_path)
        assert isinstance(logger, logging.Logger)

    def test_logger_error(self):
        with pytest.raises(TypeError):
            get_logger("test_error", 1)

    def test_logger_not_saving(self, caplog):
        get_logger("test_case_no_save", None)
        assert caplog.record_tuples == [
            ("test_case_no_save", logging.WARNING, "Logger not saving to file")
        ], "Wrong logging message"
