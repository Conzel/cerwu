import os
from typing import Any
import psutil
import torch


class Logger:
    levels = ["none", "info", "performance", "debug"]
    _instance = None

    def __init__(self, level: str = "info", flush: bool = False):
        self.has_gpu = torch.cuda.is_available()
        self.level = self.level_str_to_int(level)
        self.flush = flush

    @classmethod
    def init(cls, level: str = "info", flush: bool = False, exist_ok: bool = False):
        """Initialize the logger before first use."""
        if cls._instance is None:
            cls._instance = cls(level, flush)
        else:
            if exist_ok:
                print("Logger is already initialized, this call is ignored.")
            else:
                raise RuntimeError("Logger is already initialized!")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            raise RuntimeError("Logger is not initialized! Call Logger.init() first.")
        return cls._instance

    def level_str_to_int(self, level: str):
        if level not in self.levels:
            raise ValueError(f"Level not recognized: {level}")
        return self.levels.index(level)

    def log(self, msg: Any, level: str = "info"):
        if not isinstance(msg, str):
            msg = str(msg)
        level_index = self.level_str_to_int(level)
        if level_index <= self.level:
            print(msg, flush=self.flush)

    @staticmethod
    def bold(s: str):
        return "\033[1m" + s + "\033[0m"

    def mem_info(self):
        if self.level < 2:
            return
        ram = psutil.virtual_memory()
        ram_total = ram.total / 1e9
        ram_used = ram.used / 1e9
        ram_free = ram.available / 1e9
        process = psutil.Process(os.getpid())
        ram_allocated_by_app = process.memory_info().vms / 1e9  # Convert to GB
        ram_used_by_app = process.memory_info().rss / 1e9  # Convert to GB

        print(
            f"{self.bold('RAM')}:     {ram_used:.2f} GB / {ram_total:.2f} GB (Free: {ram_free:.2f} GB)",
            flush=self.flush,
        )
        print(
            f"{self.bold('Process')}: {ram_used_by_app:.2f} GB used, {ram_allocated_by_app:.2f} GB allocated",
            flush=self.flush,
        )

        if self.has_gpu:
            gpu_id = 0
            gpu_props = torch.cuda.get_device_properties(gpu_id)
            gpu_total = gpu_props.total_memory / 1e9
            gpu_used = torch.cuda.memory_allocated(gpu_id) / 1e9
            gpu_free = gpu_total - gpu_used
            print(
                f"{self.bold('GPU')}:     {gpu_used:.2f} GB / {gpu_total:.2f} GB (Free: {gpu_free:.2f} GB)",
                flush=self.flush,
            )
