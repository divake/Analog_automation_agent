"""
Interface to Cadence simulator for TORA-AICircuit integration.
Handles netlist generation, simulation execution, and results parsing.
Uses existing Simulation module from AICircuit.
"""

import os
import subprocess
import logging
import re
import json
import sys
from typing import Dict, Any, List, Tuple

# Add the parent directory to the path to import from Simulation
sys.path.append("/ssd_4TB/divake/AICircuit")
from Simulation.simulator import Simulator as AICircuitSimulator
from Simulation.utils.param import get_circ_params, get_circ_path

class CadenceSimulator:
    """
    Interface to Cadence simulator for TORA-AICircuit integration.
    Handles netlist generation, simulation execution, and results parsing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulator interface with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.circuit_type = config["circuit"]["type"]
        
        # Define paths based on provided information
        self.netlist_path = f"/ssd_4TB/divake/AICircuit/Simulation/Netlists/{self.circuit_type}"
        self.ocean_path = f"/ssd_4TB/divake/AICircuit/Simulation/Ocean/{self.circuit_type}"
        self.model_path = "/ssd_4TB/divake/AICircuit/Simulation/Model"
        
        # Create output directory if it doesn't exist
        self.output_dir = f"/ssd_4TB/divake/AICircuit/tora/simulation_results/{self.circuit_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get circuit parameters from existing configuration
        self.circuit_params = get_circ_params(self.circuit_type)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Store the configuration
        self.config = config
        
        # Initialize the AICircuit simulator
        self.initialize_simulator()
        
    def initialize_simulator(self):
        """
        Initialize the AICircuit simulator with the correct paths.
        """
        # Get circuit path from existing utility
        circuit_path = get_circ_path(self.circuit_type)
        
        # We'll use None for circuit_path_docker as it's not used anymore
        circuit_path_docker = None
        
        # Use the Ocean script file name from config or default to oceanScript.ocn
        ocean_filename = self.config.get("simulation", {}).get("ocean_script", "oceanScript.ocn")
        
        # Create the AICircuit simulator
        self.ai_circuit_simulator = AICircuitSimulator(
            circuit_path, 
            circuit_path_docker, 
            self.circuit_params, 
            None,  # We'll set params_path later in simulate()
            ocean_filename
        )
        
        self.logger.info(f"Initialized simulator for {self.circuit_type}")
        
    def generate_netlist(self, parameters: Dict[str, float], specs: Dict[str, float]) -> str:
        """
        Prepare simulation by updating circuit parameters.
        
        Args:
            parameters: Circuit parameters
            specs: Performance specifications
            
        Returns:
            Path to the generated netlist file
        """
        # Update the circuit parameters with the new values
        for name, value in parameters.items():
            if name in self.circuit_params:
                self.circuit_params[name] = value
        
        # Create a unique simulation ID
        simulation_id = hex(hash(frozenset(parameters.items())))[-8:]
        
        # The netlist is already in the Cadence format, so we just need to
        # alter the circuit parameters using the AICircuit simulator's alter_circ_param function
        
        # We'll use the circuit definition path from the AICircuit simulator
        circuit_def = self.ai_circuit_simulator.circuit_def
        
        # Update the circuit parameter file
        from Simulation.simulator import alter_circ_param
        alter_circ_param(self.circuit_params, circuit_def)
        
        self.logger.info(f"Generated circuit parameters for simulation ID: {simulation_id}")
        
        # Return the simulation ID for reference in other methods
        return simulation_id
    
    def run_simulation(self, simulation_id: str) -> str:
        """
        Run Cadence simulation using the generated netlist.
        
        Args:
            simulation_id: Unique simulation ID
            
        Returns:
            Path to the simulation results directory
        """
        results_dir = f"{self.output_dir}/results_{simulation_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Run the simulation using the AICircuit simulator
        self.logger.info(f"Running simulation for {self.circuit_type}")
        
        try:
            # Run the simulation
            self.ai_circuit_simulator.run_sim()
            
            # Copy the results to our results directory
            import shutil
            
            # Copy the results.txt file if it exists
            results_txt = f"{self.ai_circuit_simulator.circuit_path}/results.txt"
            if os.path.exists(results_txt):
                shutil.copy(results_txt, f"{results_dir}/results.txt")
                
            # Copy any other relevant files
            for file in ["ac.csv", "dc.csv", "bw.csv"]:
                src_file = f"{self.ai_circuit_simulator.circuit_path}/{file}"
                if os.path.exists(src_file):
                    shutil.copy(src_file, f"{results_dir}/{file}")
                
            self.logger.info(f"Simulation completed successfully")
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            # Save error output
            with open(f"{results_dir}/error.log", 'w') as f:
                f.write(f"Error: {str(e)}\n")
            raise RuntimeError(f"Simulation failed: {e}")
        
        return results_dir
    
    def parse_results(self, results_dir: str) -> Dict[str, float]:
        """
        Parse the simulation results to extract performance metrics.
        
        Args:
            results_dir: Path to the simulation results directory
            
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info(f"Parsing simulation results from: {results_dir}")
        
        # We'll use the AICircuit simulator's get_results functionality, but
        # we need to adapt it to work with our output directory
        
        # Create a temporary simulator just for parsing results
        temp_simulator = AICircuitSimulator(
            self.ai_circuit_simulator.circuit_path,
            None,
            self.circuit_params,
            None,
            self.ai_circuit_simulator.circuit_def.split('/')[-1]
        )
        
        # Clear any existing results
        temp_simulator.sim_results = []
        
        # Create a temporary results file that the get_results method can parse
        results_txt = f"{results_dir}/results.txt"
        if not os.path.exists(results_txt):
            self.logger.warning(f"Results file not found at {results_txt}")
            return {}
        
        # Make the results.txt file available to the AICircuit simulator
        import shutil
        shutil.copy(results_txt, f"{temp_simulator.circuit_path}/results.txt")
        
        # Parse the results
        temp_simulator.get_results()
        
        # Check if we have any results
        if not temp_simulator.sim_results:
            self.logger.warning("No results were parsed from the simulation")
            return {}
            
        # Format the metrics
        metrics = temp_simulator.sim_results[0].copy()
        
        # Remove ground truth and error metrics if present
        clean_metrics = {}
        for key, value in metrics.items():
            if 'Error_' not in key and 'GroundTruth' not in key:
                clean_metrics[key] = value
        
        # Save the parsed metrics to a file for reference
        with open(f"{results_dir}/metrics.json", 'w') as f:
            json.dump(clean_metrics, f, indent=2)
            
        self.logger.info(f"Parsed metrics: {clean_metrics}")
        return clean_metrics
    
    def simulate(self, parameters: Dict[str, float], specs: Dict[str, float]) -> Dict[str, float]:
        """
        Complete simulation pipeline: generate netlist, run simulation, parse results.
        
        Args:
            parameters: Circuit parameters
            specs: Performance specifications
            
        Returns:
            Dictionary of performance metrics
        """
        # Generate netlist and get simulation ID
        simulation_id = self.generate_netlist(parameters, specs)
        
        # Run the simulation
        results_dir = self.run_simulation(simulation_id)
        
        # Parse the results
        metrics = self.parse_results(results_dir)
        
        return metrics 