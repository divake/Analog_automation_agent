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
import pandas as pd
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Import modules
from utils import setup_logging, parse_specs_string, ensure_directory, calculate_relative_error
from reasoning import ReasoningModule
from simulator import CadenceSimulator
from CSVA.model import CircuitModel
from tora_visualization import ToraVisualizer

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
    parser.add_argument("--test-csv", type=str, 
                       default="/ssd_4TB/divake/AICircuit/Dataset/CSVA/MultiLayerPerceptron/test.csv", 
                       help="Path to test CSV file containing ground truth data")
    parser.add_argument("--num-samples", type=int, default=10, 
                      help="Number of samples to process from the test CSV (default: 10, -1 for all)")
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
        
        # Initialize visualizer
        logger.info("Initializing visualization module")
        visualizer = ToraVisualizer(output_dir)
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        sys.exit(1)
    
    # Check if we should use the test CSV file
    if os.path.exists(args.test_csv):
        logger.info(f"Using test data from {args.test_csv}")
        run_with_test_csv(args.test_csv, args.num_samples, config, model, reasoning, simulator, visualizer, output_dir)
    else:
        # Parse specifications from command line
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
        
        # Start the iterative refinement process with a single spec
        run_tora_process(config, specs, model, reasoning, simulator, visualizer, output_dir)
    
    # Save final visualization results
    visualizer.save_final_results()

def run_with_test_csv(csv_path: str, 
                     num_samples: int, 
                     config: Dict[str, Any],
                     model: CircuitModel,
                     reasoning: ReasoningModule,
                     simulator: CadenceSimulator,
                     visualizer: ToraVisualizer,
                     output_dir: str):
    """
    Run the TORA process using test data from a CSV file.
    
    Args:
        csv_path: Path to the test CSV file
        num_samples: Number of samples to process (-1 for all)
        config: Configuration dictionary
        model: Circuit model for parameter prediction
        reasoning: Reasoning module for explanations and refinements
        simulator: Simulator for running circuit simulations
        visualizer: Visualization module for plotting results
        output_dir: Output directory for results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load the test CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        
        # Limit number of samples if specified
        if num_samples > 0:
            df = df.head(num_samples)
            logger.info(f"Processing {len(df)} samples")
        
        # Process each sample
        for i, row in df.iterrows():
            logger.info(f"Processing sample {i+1}/{len(df)}")
            
            # Extract ground truth parameters and specs
            ground_truth = {
                "VDD": float(row["VDD"]),
                "Vgate": float(row["Vgate"]),
                "Wn": float(row["Wn"]),
                "Rd": float(row["Rd"]),
                "Bandwidth": float(row["Bandwidth"]),
                "PowerConsumption": float(row["PowerConsumption"]),
                "VoltageGain": float(row["VoltageGain"])
            }
            
            # Create specifications for the model
            specs = {
                "bandwidth": ground_truth["Bandwidth"],
                "power": ground_truth["PowerConsumption"],
                "gain": ground_truth["VoltageGain"]
            }
            
            logger.info(f"Sample {i+1} specs: {specs}")
            logger.info(f"Sample {i+1} ground truth parameters: {ground_truth}")
            
            # Run TORA process with these specifications
            results = run_tora_process_with_visualization(
                config, specs, ground_truth, model, reasoning, simulator, visualizer, output_dir, i
            )
            
            # Save individual sample results
            with open(f"{output_dir}/sample_{i+1}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Brief pause to allow visualization to update
            time.sleep(0.5)
        
        logger.info(f"Completed processing {len(df)} samples")
        
    except Exception as e:
        logger.error(f"Error processing test CSV: {e}")
        raise

def run_tora_process_with_visualization(
    config: Dict[str, Any], 
    specs: Dict[str, float],
    ground_truth: Dict[str, float],
    model: CircuitModel,
    reasoning: ReasoningModule,
    simulator: CadenceSimulator,
    visualizer: ToraVisualizer,
    output_dir: str,
    sample_id: int = 0):
    """
    Run the TORA iterative refinement process with visualization updates.
    
    Args:
        config: Configuration dictionary
        specs: Performance specifications
        ground_truth: Ground truth parameters and metrics
        model: Circuit model for parameter prediction
        reasoning: Reasoning module for explanations and refinements
        simulator: Simulator for running circuit simulations
        visualizer: Visualization module for plotting results
        output_dir: Output directory for results
        sample_id: Sample ID for visualization
    
    Returns:
        Dictionary containing all results of the TORA process
    """
    logger = logging.getLogger(__name__)
    circuit_type = config["circuit"]["type"]
    max_iterations = config["iteration"]["max_iterations"]
    convergence_threshold = config["iteration"]["convergence_threshold"]
    
    # Get absolute convergence threshold from config, or use a default
    absolute_threshold = config["iteration"].get("absolute_threshold", 0.1)
    min_iterations = config["iteration"].get("min_iterations", 1)
    
    # Dictionary to store all results
    results = {
        "specifications": specs,
        "ground_truth": ground_truth,
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
        
        # Update visualization with MLP predictions and simulation results
        visualizer.update(
            sample_id=sample_id,
            ground_truth=ground_truth,
            mlp_prediction=parameters,
            tora_prediction=parameters,  # Initially, TORA prediction is the same as MLP
            mlp_metrics=simulation_results,
            tora_metrics=simulation_results
        )
    except Exception as e:
        logger.error(f"Error during initial simulation: {e}")
        simulation_results = {"error": str(e)}
        results["iterations"][0]["simulation_error"] = str(e)
        
        # Can't continue with refinement if simulation failed
        logger.error("Cannot continue with refinement due to simulation error")
        
        # Save results so far
        with open(f"{output_dir}/sample_{sample_id}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
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
        
        # Log performance details
        logger.info(f"Performance metric: {performance:.6f} (higher is better)")
        if best_performance != float('-inf'):
            perf_change = ((performance - best_performance) / abs(best_performance)) * 100 if best_performance != 0 else float('inf')
            logger.info(f"Change from best: {perf_change:.2f}%")
        
        # Check if this is the best result so far
        if performance > best_performance:
            logger.info("Found better parameters")
            
            # Calculate improvement percentage BEFORE updating best_performance
            prev_best_performance = best_performance
            improvement = (performance - prev_best_performance) / abs(prev_best_performance) if prev_best_performance != 0 else float('inf')
            logger.info(f"Improvement: {improvement*100:.2f}%")
            
            # Now update best values
            best_performance = performance
            best_parameters = current_parameters.copy()
            best_metrics = simulation_results.copy()
            
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
            
            # Update visualization with the latest TORA results
            visualizer.update(
                sample_id=sample_id,
                ground_truth=ground_truth,
                mlp_prediction=parameters,  # Original MLP prediction
                tora_prediction=current_parameters,  # Latest TORA refined parameters
                mlp_metrics=results["iterations"][0]["simulation_results"],  # Original MLP metrics
                tora_metrics=simulation_results  # Latest TORA metrics
            )
            
            # Check for convergence
            if iteration >= min_iterations and abs(improvement) < convergence_threshold:
                logger.info(f"Converged within relative improvement threshold ({convergence_threshold})")
                break
                
            # Additional check: if total error is already very small, we can stop
            if abs(best_performance) < absolute_threshold:
                logger.info(f"Converged within absolute error threshold ({absolute_threshold})")
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
    
    # One final visualization update with the best results
    visualizer.update(
        sample_id=sample_id,
        ground_truth=ground_truth,
        mlp_prediction=parameters,  # Original MLP prediction
        tora_prediction=best_parameters,  # Best TORA parameters
        mlp_metrics=results["iterations"][0]["simulation_results"],  # Original MLP metrics
        tora_metrics=best_metrics  # Best TORA metrics
    )
    
    # Save results
    with open(f"{output_dir}/sample_{sample_id}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/sample_{sample_id}_results.json")
    
    return results

def run_tora_process(config: Dict[str, Any], 
                    specs: Dict[str, float],
                    model: CircuitModel,
                    reasoning: ReasoningModule,
                    simulator: CadenceSimulator,
                    visualizer: ToraVisualizer,
                    output_dir: str):
    """
    Wrapper for the original TORA process without ground truth (single sample).
    
    Args:
        config: Configuration dictionary
        specs: Performance specifications
        model: Circuit model for parameter prediction
        reasoning: Reasoning module for explanations and refinements
        simulator: Simulator for running circuit simulations
        visualizer: Visualization module
        output_dir: Output directory for results
    """
    logger = logging.getLogger(__name__)
    
    # Create dummy ground truth from specs
    ground_truth = {
        "VDD": 1.5,
        "Vgate": 0.7,
        "Wn": 5e-6,
        "Rd": 1000,
        "Bandwidth": specs.get("bandwidth", 100e6),
        "PowerConsumption": specs.get("power", 1e-3),
        "VoltageGain": specs.get("gain", 10)
    }
    
    # Run the process with visualization
    results = run_tora_process_with_visualization(
        config, specs, ground_truth, model, reasoning, simulator, visualizer, output_dir
    )
    
    # Save overall results
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/results.json")

if __name__ == "__main__":
    main() 