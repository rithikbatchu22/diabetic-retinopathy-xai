from src.utils.seed import seed_everything
from src.utils.logger import get_logger

seed_everything(42)
logger = get_logger(log_file="reports/generated/run.log")
logger.info("Phase 0 sanity check passed.")
print("OK")