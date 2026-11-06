import os
import sys
import logging
import traceback
from pathlib import Path
from omegaconf import DictConfig
import hydra
from loguru import logger
import dotenv
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.configuration import migrate_legacy_config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv.load_dotenv()

def _run(config:DictConfig):
    config = migrate_legacy_config(config)
    # Configure loguru log level based on config
    log_level = "DEBUG" if config.executor.debug else "INFO"
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    for logger_name in logging.root.manager.loggerDict:
        if "zotero_arxiv_daily" in logger_name:
            continue
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if config.executor.debug:
        logger.info("Debug mode is enabled")
    
    executor = Executor(config)
    executor.run()

@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(config:DictConfig):
    try:
        _run(config)
    except Exception as exc:
        # Provider and interpolation exceptions can contain credentials or full
        # configuration values. Keep Actions output safe even in debug mode.
        # Log only code locations and exception types, never source lines,
        # frame locals, exception messages, prompts, or configuration values.
        frames = list(traceback.walk_tb(exc.__traceback__))
        frame, line = frames[-1]
        location = f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{line}"
        cause = exc.__cause__
        if cause is not None and cause.__traceback__ is not None:
            cause_frame, cause_line = list(traceback.walk_tb(cause.__traceback__))[-1]
            cause_location = f"{Path(cause_frame.f_code.co_filename).name}:{cause_frame.f_code.co_name}:{cause_line}"
            location += f"; caused by {type(cause).__name__} at {cause_location}"
        logger.error("Run failed ({}) at {}. Check required settings and service availability; details withheld.", type(exc).__name__, location)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
