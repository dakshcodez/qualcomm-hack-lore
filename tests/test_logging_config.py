import logging

from pc.api.logging_config import LOGGER_NAME, configure_logging, get_request_logger, new_request_id


def test_configure_logging_attaches_file_and_console_handlers(tmp_path):
    log_file = tmp_path / "logs" / "lore.log"
    logger = configure_logging(log_file=log_file)

    assert logger.name == LOGGER_NAME
    assert len(logger.handlers) == 2
    assert log_file.parent.is_dir()


def test_configure_logging_is_idempotent(tmp_path):
    log_file = tmp_path / "lore.log"
    configure_logging(log_file=log_file)
    logger = configure_logging(log_file=log_file)
    assert len(logger.handlers) == 2


def test_configure_logging_without_file_only_attaches_console_handler():
    logger = configure_logging(log_file=None)
    assert len(logger.handlers) == 1


def test_new_request_id_is_unique():
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100


def test_log_messages_are_written_to_the_configured_file(tmp_path):
    log_file = tmp_path / "lore.log"
    configure_logging(log_file=log_file, level=logging.DEBUG)

    request_logger = get_request_logger(request_id="req-abc123")
    request_logger.info("embedding completed in %.2fms", 12.34)

    contents = log_file.read_text()
    assert "req-abc123" in contents
    assert "embedding completed in 12.34ms" in contents
    assert "INFO" in contents


def test_plain_logger_without_request_id_falls_back_to_dash(tmp_path):
    log_file = tmp_path / "lore.log"
    configure_logging(log_file=log_file, level=logging.DEBUG)

    logging.getLogger(LOGGER_NAME).warning("NPU fallback detected")

    contents = log_file.read_text()
    assert "[-]" in contents
    assert "NPU fallback detected" in contents


def test_get_request_logger_generates_id_when_omitted():
    adapter = get_request_logger()
    assert isinstance(adapter.extra["request_id"], str)
    assert len(adapter.extra["request_id"]) > 0


def test_caplog_sees_records_via_propagation(tmp_path, caplog):
    log_file = tmp_path / "lore.log"
    configure_logging(log_file=log_file, level=logging.DEBUG)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        get_request_logger(request_id="req-xyz").info("search latency: %dms", 5)

    assert "search latency: 5ms" in caplog.text
