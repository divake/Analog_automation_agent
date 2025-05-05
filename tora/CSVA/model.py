"""
CSVA-specific model interface for TORA-AICircuit.
Interfaces with AICircuit ML models for CSVA parameter prediction.
"""

import os
import numpy as np
import torch
import pickle
import logging
import pandas as pd
import sys
from typing import Dict, Any, List, Tuple

# Add parent directory to the path for importing
sys.path.append("/ssd_4TB/divake/AICircuit")

class CircuitModel:
    """
    Interface to AICircuit ML model for CSVA circuit.
    Handles parameter prediction based on performance specifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the circuit model with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        # Initialize logger first so it's available for all other initialization methods
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.model_type = config["model"]["type"]
        
        # Override the weights path with the actual path to the pretrained model
        self.weights_path = "/ssd_4TB/divake/AICircuit/Dataset/CSVA/MultiLayerPerceptron/model.pkl"
        self.circuit_knowledge_path = "/ssd_4TB/divake/AICircuit/Dataset/CSVA/CSVA.csv"
        
        self.input_dim = config["model"]["input_dim"]
        self.output_dim = config["model"]["output_dim"]
        
        # Load circuit knowledge to get parameter ranges
        self.circuit_knowledge = self._load_circuit_knowledge()
        
        # Parameter names and specification names for CSVA
        # These need to match the columns in the dataset and the model inputs/outputs
        self.param_names = ["VDD", "Vgate", "Wn", "Rd"]
        self.spec_names = ["Bandwidth", "PowerConsumption", "VoltageGain"]
        
        # Determine parameter ranges from circuit knowledge
        self.param_ranges = self._get_param_ranges_from_knowledge()
        self.spec_ranges = self._get_spec_ranges_from_knowledge()
        
        # Initialize the ML model
        self._initialize_model()
    
    def _load_circuit_knowledge(self):
        """Load circuit knowledge from CSV file"""
        try:
            return pd.read_csv(self.circuit_knowledge_path)
        except Exception as e:
            self.logger.error(f"Error loading circuit knowledge: {e}")
            return None
    
    def _get_param_ranges_from_knowledge(self):
        """Get parameter ranges from circuit knowledge"""
        if self.circuit_knowledge is not None:
            # Calculate min/max for each parameter
            param_ranges = {}
            for param in self.param_names:
                if param in self.circuit_knowledge:
                    min_val = self.circuit_knowledge[param].min()
                    max_val = self.circuit_knowledge[param].max()
                    param_ranges[param] = (min_val, max_val)
                else:
                    # Default ranges if parameter not found
                    param_ranges[param] = (0.0, 1.0)
            return param_ranges
        else:
            # Default ranges if knowledge not available
            return {
                "VDD": (1.0, 1.8),
                "Vgate": (0.4, 0.8),
                "Wn": (1e-6, 10e-6),
                "Rd": (500, 3000)
            }
    
    def _get_spec_ranges_from_knowledge(self):
        """Get specification ranges from circuit knowledge"""
        if self.circuit_knowledge is not None:
            # Calculate min/max for each specification
            spec_ranges = {}
            for spec in self.spec_names:
                if spec in self.circuit_knowledge:
                    min_val = self.circuit_knowledge[spec].min()
                    max_val = self.circuit_knowledge[spec].max()
                    spec_ranges[spec] = (min_val, max_val)
                else:
                    # Default ranges if specification not found
                    spec_ranges[spec] = (0.0, 1.0)
            return spec_ranges
        else:
            # Default ranges if knowledge not available
            return {
                "Bandwidth": (50e6, 500e6),
                "PowerConsumption": (100e-6, 2000e-6),
                "VoltageGain": (0, 15)
            }
    
    def _initialize_model(self):
        """Initialize the appropriate ML model based on configuration"""
        if self.model_type == "MultiLayerPerceptron":
            try:
                # Load the pretrained model
                if os.path.exists(self.weights_path):
                    self.logger.info(f"Loading model from: {self.weights_path}")
                    
                    with open(self.weights_path, 'rb') as f:
                        self.model = pickle.load(f)
                    
                    self.logger.info("Model loaded successfully")
                else:
                    self.logger.warning(f"Model not found at: {self.weights_path}")
                    self.logger.warning("Using a simulated model for demonstration")
                    self.model = None
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                self.logger.warning("Using a simulated model for demonstration")
                self.model = None
        else:
            self.logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _map_api_specs_to_model_specs(self, specs: Dict[str, float]) -> Dict[str, float]:
        """
        Map API specification names to model specification names.
        
        Args:
            specs: Dictionary with API specification names
            
        Returns:
            Dictionary with model specification names
        """
        spec_mapping = {
            "gain": "VoltageGain",
            "bandwidth": "Bandwidth",
            "power": "PowerConsumption"
        }
        
        model_specs = {}
        for api_name, value in specs.items():
            if api_name in spec_mapping:
                model_name = spec_mapping[api_name]
                model_specs[model_name] = value
            else:
                model_specs[api_name] = value
        
        return model_specs
    
    def _normalize_specs(self, specs: Dict[str, float]) -> np.ndarray:
        """
        Normalize the specifications for the ML model.
        
        Args:
            specs: Dictionary of specifications
            
        Returns:
            Normalized specifications as numpy array
        """
        # First map API spec names to model spec names
        model_specs = self._map_api_specs_to_model_specs(specs)
        
        normalized = []
        
        for name in self.spec_names:
            if name in model_specs:
                value = model_specs[name]
                min_val, max_val = self.spec_ranges[name]
                
                # Clip value to range
                value = max(min_val, min(max_val, value))
                
                # Normalize
                norm_value = (value - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
                normalized.append(norm_value)
            else:
                self.logger.warning(f"Specification {name} not provided, using default")
                normalized.append(0.5)  # Default normalized value
        
        return np.array(normalized, dtype=np.float32).reshape(1, -1)
    
    def _denormalize_params(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """
        Denormalize the parameters from the ML model.
        
        Args:
            normalized_params: Normalized parameters as numpy array
            
        Returns:
            Dictionary of denormalized parameters
        """
        params = {}
        
        for i, name in enumerate(self.param_names):
            if i < normalized_params.shape[1]:
                norm_value = normalized_params[0, i]
                min_val, max_val = self.param_ranges[name]
                
                # Clip to ensure values are within range
                norm_value = max(0.0, min(1.0, norm_value))
                
                # Denormalize
                value = min_val + norm_value * (max_val - min_val)
                
                params[name] = value
        
        return params
    
    def _map_model_params_to_simulator_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Map model parameter names to simulator parameter names.
        
        Args:
            params: Dictionary with model parameter names
            
        Returns:
            Dictionary with simulator parameter names
        """
        # For CSVA, this is mostly a pass-through since the names should match
        # the simulator's expectation. But we can add any necessary conversions here.
        
        # Make a copy to avoid modifying the original
        simulator_params = params.copy()
        
        return simulator_params
    
    def predict(self, specs: Dict[str, float]) -> Dict[str, float]:
        """
        Predict circuit parameters based on performance specifications.
        
        Args:
            specs: Dictionary of performance specifications
            
        Returns:
            Dictionary of predicted circuit parameters
        """
        self.logger.info(f"Predicting parameters for specs: {specs}")
        
        # Normalize specifications
        normalized_specs = self._normalize_specs(specs)
        
        if self.model is not None:
            try:
                # Use the pretrained model to predict parameters
                normalized_params = self.model.predict(normalized_specs)
                
                # Convert to numpy array if it isn't already
                if not isinstance(normalized_params, np.ndarray):
                    normalized_params = np.array(normalized_params).reshape(1, -1)
                
                # Denormalize to get actual parameter values
                params = self._denormalize_params(normalized_params)
                
                # Map to simulator parameter names
                simulator_params = self._map_model_params_to_simulator_params(params)
                
                self.logger.info(f"Predicted parameters: {simulator_params}")
                return simulator_params
            except Exception as e:
                self.logger.error(f"Error during parameter prediction with model: {e}")
                # Fall back to heuristic prediction
                return self._heuristic_predict(specs)
        else:
            # No model available, use heuristic-based prediction
            return self._heuristic_predict(specs)
    
    def _heuristic_predict(self, specs: Dict[str, float]) -> Dict[str, float]:
        """
        Fallback heuristic-based parameter prediction when the model isn't available.
        
        Args:
            specs: Dictionary of performance specifications
            
        Returns:
            Dictionary of predicted circuit parameters
        """
        self.logger.info("Using heuristic-based prediction")
        
        # Map API spec names to model spec names
        model_specs = self._map_api_specs_to_model_specs(specs)
        
        try:
            # Extract specifications
            voltage_gain = model_specs.get("VoltageGain", 10.0)
            bandwidth = model_specs.get("Bandwidth", 100e6)
            power = model_specs.get("PowerConsumption", 500e-6)
            
            # Fixed supply voltage for CSVA
            vdd = 1.2
            
            # Simplified heuristic-based prediction
            # These formulas are based on simplified circuit analysis
            
            # Set Vgate based on desired tradeoff between gain and bandwidth
            vgate = 0.6 + 0.2 * (voltage_gain / 15.0)
            
            # Determine transistor width based on bandwidth and power
            wn = (power * 1e6) / (100.0 * bandwidth / 1e8)
            wn = max(3e-6, min(10e-6, wn))  # Limit to reasonable range
            
            # Determine drain resistor based on gain
            rd = 500.0 + 1500.0 * (voltage_gain / 15.0)
            rd = max(500, min(3000, rd))  # Limit to reasonable range
            
            # Construct parameter dictionary
            params = {
                "VDD": vdd,
                "Vgate": vgate,
                "Wn": wn,
                "Rd": rd
            }
            
            self.logger.info(f"Heuristic-predicted parameters: {params}")
            return params
            
        except Exception as e:
            self.logger.error(f"Error during heuristic parameter prediction: {e}")
            
            # Fallback to default parameters
            self.logger.warning("Using default parameters")
            return {
                "VDD": 1.2,
                "Vgate": 0.7,
                "Wn": 5e-6,
                "Rd": 1000.0
            } 