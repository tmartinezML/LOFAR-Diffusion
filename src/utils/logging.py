import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and level. If the logger already has
    handlers, clear and reset them.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level, by default logging.INFO

    Returns
    -------
    logging.Logger
        The logger with the given name and level.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # Check if the logger already has handlers
        logger.handlers.clear()  # Clear the default handlers
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s (%(name)s): %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def show_dl_progress(block_num, block_size, total_size):
    global pbar, last_loaded
    if pbar is None:
        pbar = tqdm(total=total_size, unit="Bytes", unit_scale=True)

    downloaded = block_num * block_size
    increment = downloaded - last_loaded
    last_loaded = downloaded
    if downloaded < total_size:
        pbar.update(increment)
    else:
        pbar.close()
        pbar, last_loaded = None, 0