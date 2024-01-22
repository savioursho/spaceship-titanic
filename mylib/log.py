import contextlib
import time
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Optional


@dataclass
class Timer:
    logger: Optional[Logger] = None

    def logging_func(self) -> Callable:
        if self.logger is None:
            return print
        else:
            return self.logger.info

    @contextlib.contextmanager
    def measure(self, name: str):
        logging_func = self.logging_func()
        start = time.time()
        logging_func(f"[{name}] start.")
        yield
        self.duration = time.time() - start
        logging_func(f"[{name}] done in {self.duration:.0f} s.")
