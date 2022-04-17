import logging
from rich.logging import RichHandler


class Logger:

    FORMAT = "%(message)s"

    def __init__(self, name: str = "autonomy_utils"):
        self.log = logging.getLogger(name)

        # File handler
        # f_handler = logging.FileHandler("./assignment4/logs/icp_log.log", mode="w")
        # f_handler.setLevel("INFO")
        # f_handler.setFormatter(logging.Formatter(fmt=Logger.FORMAT, datefmt="[%X]"))
        # log.addHandler(f_handler)

        # Simple Stream handler
        # s_handler = logging.StreamHandler()
        # s_handler.setLevel("INFO")
        # s_handler.setFormatter(logging.Formatter(fmt=Logger.FORMAT, datefmt="[%X]"))
        # self.log.addHandler(s_handler)

        # Rich stream handler
        r_handler = RichHandler(rich_tracebacks=True)
        r_handler.setLevel("DEBUG")
        r_handler.setFormatter(logging.Formatter(fmt=Logger.FORMAT, datefmt="[%X]"))
        self.log.addHandler(r_handler)
        self.log.setLevel("DEBUG")
