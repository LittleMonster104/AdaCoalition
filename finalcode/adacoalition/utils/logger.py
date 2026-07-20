"""Logging utilities"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Logger for experiment metrics and results"""
    
    def __init__(self, experiment_name: str, output_dir: str):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(experiment_name, str(self.output_dir))
        
        self.metrics = []
    
    def log_metric(self, metric_name: str, value: float, step: int = None):
        """Log a metric value"""
        entry = {
            'metric': metric_name,
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics.append(entry)
        
        msg = f"{metric_name}: {value:.4f}"
        if step is not None:
            msg = f"Step {step} - {msg}"
        
        self.logger.info(msg)
    
    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.logger.info("=" * 50)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)
    
    def save_results(self, results: dict):
        """Save experiment results to file"""
        import json
        import numpy as np
        import pandas as pd
        
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(val) for key, val in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            return obj
        
        results_file = self.output_dir / f"{self.experiment_name}_results.json"
        
        # Convert results to JSON-serializable format
        serializable_results = convert_to_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
