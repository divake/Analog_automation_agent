"""
Utility functions for TORA-AICircuit.
"""

import os
import logging
import yaml
from typing import Dict, Any, List, Tuple

def setup_logging(level: str, log_file: str, console: bool = True) -> None:
    """
    Set up logging with file and console handlers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console: Whether to log to console
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Create console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def parse_specs_string(specs_str: str) -> Dict[str, float]:
    """
    Parse comma-separated key=value pairs into a dictionary of specifications.
    
    Args:
        specs_str: String of comma-separated key=value pairs
        
    Returns:
        Dictionary of specifications
    """
    specs = {}
    
    if not specs_str:
        return specs
    
    pairs = specs_str.split(',')
    for pair in pairs:
        if '=' in pair:
            key, value_str = pair.split('=', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            # Try to parse the value
            try:
                # Check for scientific notation
                if 'e' in value_str.lower():
                    value = float(value_str)
                # Check for decimal point
                elif '.' in value_str:
                    value = float(value_str)
                # Otherwise, try to parse as int first, then fall back to float
                else:
                    try:
                        value = int(value_str)
                    except ValueError:
                        value = float(value_str)
                
                specs[key] = value
            except ValueError:
                # If parsing fails, keep the string value
                specs[key] = value_str
    
    return specs

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def format_number(value: float) -> str:
    """
    Format a number with appropriate units.
    
    Args:
        value: Numeric value
        
    Returns:
        Formatted string with units
    """
    # Handle different ranges of values
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f} G"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f} M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f} k"
    elif abs(value) >= 1:
        return f"{value:.2f}"
    elif abs(value) >= 1e-3:
        return f"{value*1e3:.2f} m"
    elif abs(value) >= 1e-6:
        return f"{value*1e6:.2f} µ"
    elif abs(value) >= 1e-9:
        return f"{value*1e9:.2f} n"
    elif abs(value) >= 1e-12:
        return f"{value*1e12:.2f} p"
    else:
        return f"{value:.2e}"

def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The directory path
    """
    os.makedirs(path, exist_ok=True)
    return path

def calculate_relative_error(predicted: Dict[str, float], 
                            actual: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate relative error between predicted and actual values.
    
    Args:
        predicted: Dictionary of predicted values
        actual: Dictionary of actual values
        
    Returns:
        Dictionary of relative errors
    """
    errors = {}
    
    for key, actual_value in actual.items():
        if key in predicted and actual_value != 0:
            predicted_value = predicted[key]
            rel_error = abs((predicted_value - actual_value) / actual_value)
            errors[key] = rel_error
    
    return errors

def calculate_mean_relative_error(errors: Dict[str, float]) -> float:
    """
    Calculate mean relative error from a dictionary of errors.
    
    Args:
        errors: Dictionary of relative errors
        
    Returns:
        Mean relative error
    """
    if not errors:
        return 0.0
    
    total_error = sum(errors.values())
    return total_error / len(errors) 