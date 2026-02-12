# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-04-28

import time
from functools import wraps
from .pickt_logger import pickt_logger


logger = pickt_logger(__name__)

def measure_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        logger.info(f"Function {func.__name__} took {minutes:02d}min : {seconds:02d}sec to execute.")
        return result
    return wrapper