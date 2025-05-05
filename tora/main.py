"""
Main entry point for TORA-AICircuit.
Integrates ML models, LLM reasoning, and circuit simulation for parameter refinement.
"""

import os
import sys
import yaml
import json
import logging
import argparse
import time
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Import modules
from utils import setup_logging, parse_specs_string, ensure_directory, calculate_relative_error
from reasoning import ReasoningModule
from simulator import CadenceSimulator
from CSVA.model import CircuitModel

def main():
    """
    Main entry point for TORA-AICircuit.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TORA-AICircuit: Tool-integrated Reasoning Agent for AICircuit")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--specs", type=str, help="Performance specifications as comma-separated key=value pairs")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--circuit", type=str, default="CSVA", help="Circuit type (default: CSVA)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set circuit type in config
    config["circuit"]["type"] = args.circuit
    
    # Setup logging
    logs_dir = os.path.dirname(config["logging"]["file"])
    ensure_directory(logs_dir)
    setup_logging(config["logging"]["level"], config["logging"]["file"], config["logging"]["console"])
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting TORA-AICircuit v{config['project']['version']}")
    
    # Parse specifications
    if args.specs:
        specs = parse_specs_string(args.specs)
    else:
        # Use default specs from config
        specs = config.get("default_specs", {
            "gain": 20,
            "bandwidth": 100e6,
            "power": 1e-3
        })
    
    logger.info(f"Using specifications: {specs}")
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = int(time.time())
        output_dir = f"/ssd_4TB/divake/AICircuit/tora/results/{timestamp}"
    
    ensure_directory(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize components
    try:
        # Load ML model
        logger.info("Initializing circuit model")
        model = CircuitModel(config)
        
        # Initialize reasoning module
        logger.info("Initializing reasoning module")
        reasoning = ReasoningModule(config)
        
        # Initialize simulator
        logger.info("Initializing simulator")
        simulator = CadenceSimulator(config)
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        sys.exit(1)
    
    # Start the iterative refinement process
    run_tora_process(config, specs, model, reasoning, simulator, output_dir)

def run_tora_process(config: Dict[str, Any], 
                   specs: Dict[str, float],
                   model: CircuitModel,
                   reasoning: ReasoningModule,
                   simulator: CadenceSimulator,
                   output_dir: str):
    """
    Run the TORA iterative refinement process.
    
    Args:
        config: Configuration dictionary
        specs: Performance specifications
        model: Circuit model for parameter prediction
        reasoning: Reasoning module for explanations and refinements
        simulator: Simulator for running circuit simulations
        output_dir: Output directory for results
    """
    logger = logging.getLogger(__name__)
    circuit_type = config["circuit"]["type"]
    max_iterations = config["iteration"]["max_iterations"]
    convergence_threshold = config["iteration"]["convergence_threshold"]
    
    # Dictionary to store all results
    results = {
        "specifications": specs,
        "iterations": []
    }
    
    # Initial parameter prediction
    logger.info("Predicting initial parameters")
    parameters = model.predict(specs)
    logger.info(f"Initial parameters: {parameters}")
    
    # Get explanation for initial parameters
    logger.info("Generating explanation for initial parameters")
    explanation = reasoning.explain_parameters(circuit_type, specs, parameters)
    
    # Save initial prediction and explanation
    iteration_result = {
        "iteration": 0,
        "parameters": parameters,
        "explanation": explanation
    }
    
    results["iterations"].append(iteration_result)
    
    # Run initial simulation
    logger.info("Running initial simulation")
    try:
        simulation_results = simulator.simulate(parameters, specs)
        logger.info(f"Initial simulation results: {simulation_results}")
        
        # Add simulation results to iteration result
        results["iterations"][0]["simulation_results"] = simulation_results
    except Exception as e:
        logger.error(f"Error during initial simulation: {e}")
        simulation_results = {"error": str(e)}
        results["iterations"][0]["simulation_error"] = str(e)
        
        # Can't continue with refinement if simulation failed
        logger.error("Cannot continue with refinement due to simulation error")
        
        # Save results so far
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return
    
    # Iterative refinement loop
    current_parameters = parameters.copy()
    best_parameters = parameters.copy()
    best_metrics = simulation_results.copy()
    
    # Define a simple metric to evaluate performance
    def calculate_performance_metric(sim_results, target_specs):
        if not sim_results or "error" in sim_results:
            return float('-inf')
        
        # Convert API spec names to simulator spec names
        spec_mapping = {
            "gain": "VoltageGain",
            "bandwidth": "Bandwidth",
            "power": "PowerConsumption"
        }
        
        # Check if we have all required metrics
        required_metrics = [spec_mapping.get(spec, spec) for spec in target_specs.keys()]
        missing_metrics = [metric for metric in required_metrics if metric not in sim_results]
        
        # If any required metric is missing, return very poor performance
        if missing_metrics:
            logger.warning(f"Missing required metrics: {missing_metrics}")
            return float('-inf')
        
        total_error = 0
        
        for api_spec, target in target_specs.items():
            sim_spec = spec_mapping.get(api_spec, api_spec)
            if sim_spec in sim_results:
                achieved = sim_results[sim_spec]
                # Calculate relative error
                rel_error = abs((achieved - target) / target)
                total_error += rel_error
        
        # Return negative error (higher is better)
        return -total_error
    
    best_performance = calculate_performance_metric(simulation_results, specs)
    
    for iteration in range(1, max_iterations + 1):
        logger.info(f"Starting iteration {iteration}/{max_iterations}")
        
        # Get refinement suggestions based on simulation results
        logger.info("Generating parameter refinement suggestions")
        refinements = reasoning.suggest_refinements(circuit_type, specs, current_parameters, simulation_results)
        
        # Process refinement suggestions
        if "error" in refinements:
            logger.error(f"Error in refinement suggestions: {refinements['error']}")
            break
        
        # Apply parameter adjustments
        if "parameter_adjustments" in refinements:
            logger.info("Applying parameter adjustments")
            adjusted_parameters = current_parameters.copy()
            
            for param_name, adjustment in refinements["parameter_adjustments"].items():
                if param_name in adjusted_parameters and "suggested_value" in adjustment:
                    adjusted_parameters[param_name] = adjustment["suggested_value"]
                    logger.info(f"Adjusted {param_name}: {current_parameters[param_name]} → {adjustment['suggested_value']}")
            
            current_parameters = adjusted_parameters
        else:
            logger.warning("No parameter adjustments suggested, using current parameters")
        
        # Run simulation with adjusted parameters
        logger.info(f"Running simulation with adjusted parameters: {current_parameters}")
        try:
            simulation_results = simulator.simulate(current_parameters, specs)
            logger.info(f"Simulation results: {simulation_results}")
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            simulation_results = {"error": str(e)}
            
            # Save iteration result even though simulation failed
            iteration_result = {
                "iteration": iteration,
                "parameters": current_parameters,
                "refinements": refinements,
                "simulation_error": str(e)
            }
            
            results["iterations"].append(iteration_result)
            continue
        
        # Calculate performance metric
        performance = calculate_performance_metric(simulation_results, specs)
        
        # Check if this is the best result so far
        if performance > best_performance:
            logger.info("Found better parameters")
            best_performance = performance
            best_parameters = current_parameters.copy()
            best_metrics = simulation_results.copy()
            
            # Calculate improvement percentage
            improvement = (performance - best_performance) / abs(best_performance) if best_performance != 0 else float('inf')
            
            # Save iteration result
            iteration_result = {
                "iteration": iteration,
                "parameters": current_parameters,
                "refinements": refinements,
                "simulation_results": simulation_results,
                "performance_metric": performance,
                "improvement": improvement
            }
            
            results["iterations"].append(iteration_result)
            
            # Check for convergence
            if abs(improvement) < convergence_threshold:
                logger.info(f"Converged within threshold ({convergence_threshold})")
                break
        else:
            # Not an improvement, but still save the result
            logger.info(f"No improvement in performance: {performance} vs best {best_performance}")
            
            iteration_result = {
                "iteration": iteration,
                "parameters": current_parameters,
                "refinements": refinements,
                "simulation_results": simulation_results,
                "performance_metric": performance,
                "improvement": 0
            }
            
            results["iterations"].append(iteration_result)
            
            # Revert to best parameters for next iteration
            current_parameters = best_parameters.copy()
    
    # Final results
    logger.info(f"Best parameters: {best_parameters}")
    logger.info(f"Best metrics: {best_metrics}")
    
    # Add summary to results
    results["best_parameters"] = best_parameters
    results["best_metrics"] = best_metrics
    results["final_performance"] = best_performance
    results["completed_iterations"] = len(results["iterations"]) - 1  # Exclude initial prediction
    
    # Save results
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/results.json")

if __name__ == "__main__":
    main() 