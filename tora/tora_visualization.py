import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import logging
import time

class ToraVisualizer:
    """
    Visualization module for TORA-AICircuit.
    Shows real-time plots comparing ground truth, MLP predictions, and TORA predictions.
    Also saves results to a CSV file for later analysis.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualization module.
        
        Args:
            output_dir: Directory to save results and plots
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.columns = [
            "sample_id", 
            "VDD_GT", "Vgate_GT", "Wn_GT", "Rd_GT", 
            "Bandwidth_GT", "PowerConsumption_GT", "VoltageGain_GT",
            "VDD_MLP", "Vgate_MLP", "Wn_MLP", "Rd_MLP",
            "Bandwidth_MLP", "PowerConsumption_MLP", "VoltageGain_MLP",
            "VDD_TORA", "Vgate_TORA", "Wn_TORA", "Rd_TORA",
            "Bandwidth_TORA", "PowerConsumption_TORA", "VoltageGain_TORA",
            "VDD_MLP_PCT", "Vgate_MLP_PCT", "Wn_MLP_PCT", "Rd_MLP_PCT",
            "Bandwidth_MLP_PCT", "PowerConsumption_MLP_PCT", "VoltageGain_MLP_PCT",
            "VDD_TORA_PCT", "Vgate_TORA_PCT", "Wn_TORA_PCT", "Rd_TORA_PCT",
            "Bandwidth_TORA_PCT", "PowerConsumption_TORA_PCT", "VoltageGain_TORA_PCT"
        ]
        
        # Create CSV file and write header
        self.csv_path = os.path.join(output_dir, "results.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
        
        # Initialize plots - changed from 2x3 to 1x3 to show only performance metrics
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.tight_layout(pad=3.0)
        self.fig.suptitle("TORA-AICircuit Performance Comparison", fontsize=16)
        
        # Configure the plots
        self._configure_plots()
        
        # Save the initial plot
        self._save_current_plot(0)
        
        self.logger.info(f"Initialized visualizer, results will be saved to {self.csv_path}")
    
    def _configure_plots(self):
        """Configure the visualization plots with labels and titles."""
        # Only showing performance metrics now
        metrics = ["Bandwidth", "PowerConsumption", "VoltageGain"]
        
        # Titles for each subplot
        titles = ["Bandwidth", "Power Consumption", "Voltage Gain"]
        
        # Units for each parameter
        units = ["MHz", "mW", "V/V"]
        
        # Ensure axes is always treated as an array
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])
        
        # Configure each plot
        for i, (metric, title, unit) in enumerate(zip(metrics, titles, units)):
            if i < len(self.axes):
                ax = self.axes[i]
                ax.set_title(f"{title}")
                ax.set_xlabel("Sample ID")
                ax.set_ylabel(f"{title} ({unit})")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Create empty plots that we'll update later
                ax.plot([], [], 'bo-', label='Ground Truth', markersize=3)
                ax.plot([], [], 'ro-', label='MLP Prediction', markersize=3)
                ax.plot([], [], 'go-', label='TORA Prediction', markersize=3)
                ax.legend()
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for title
    
    def _save_current_plot(self, sample_id: int) -> None:
        """
        Save the current plot to a file.
        
        Args:
            sample_id: Current sample ID
        """
        plot_path = os.path.join(self.output_dir, f"plot_sample_{sample_id}.png")
        self.fig.savefig(plot_path, dpi=100)
        self.logger.debug(f"Saved plot to {plot_path}")
    
    def _calculate_percentage_change(self, predicted: float, ground_truth: float) -> float:
        """
        Calculate the percentage change of a prediction relative to ground truth.
        
        Args:
            predicted: Predicted value
            ground_truth: Ground truth value
            
        Returns:
            Percentage change
        """
        if ground_truth == 0:
            return float('inf') if predicted != 0 else 0.0
        return 100.0 * (predicted - ground_truth) / ground_truth
    
    def update(self, 
              sample_id: int, 
              ground_truth: Dict[str, float], 
              mlp_prediction: Dict[str, float], 
              tora_prediction: Dict[str, float],
              mlp_metrics: Dict[str, float] = None,
              tora_metrics: Dict[str, float] = None) -> None:
        """
        Update the visualization with new data and save to CSV.
        
        Args:
            sample_id: Sample ID
            ground_truth: Ground truth parameters and metrics
            mlp_prediction: MLP predicted parameters
            tora_prediction: TORA refined parameters
            mlp_metrics: Performance metrics from MLP parameters
            tora_metrics: Performance metrics from TORA parameters
        """
        # Ensure we have metrics for all three
        if mlp_metrics is None:
            mlp_metrics = {}
        if tora_metrics is None:
            tora_metrics = {}
        
        # Combine parameters and metrics for ground truth
        gt_combined = ground_truth.copy()
        
        # Prepare row data for CSV
        row_data = {"sample_id": sample_id}
        
        # Add ground truth values
        for key, value in gt_combined.items():
            row_data[f"{key}_GT"] = value
        
        # Add MLP prediction values and calculate percentage changes
        for key, gt_value in gt_combined.items():
            # Parameters
            if key in mlp_prediction:
                row_data[f"{key}_MLP"] = mlp_prediction[key]
                row_data[f"{key}_MLP_PCT"] = self._calculate_percentage_change(mlp_prediction[key], gt_value)
        
        # Add MLP metrics if available
        for key, value in mlp_metrics.items():
            row_data[f"{key}_MLP"] = value
            if key in gt_combined:
                row_data[f"{key}_MLP_PCT"] = self._calculate_percentage_change(value, gt_combined[key])
        
        # Add TORA prediction values and calculate percentage changes
        for key, gt_value in gt_combined.items():
            # Parameters
            if key in tora_prediction:
                row_data[f"{key}_TORA"] = tora_prediction[key]
                row_data[f"{key}_TORA_PCT"] = self._calculate_percentage_change(tora_prediction[key], gt_value)
        
        # Add TORA metrics if available
        for key, value in tora_metrics.items():
            row_data[f"{key}_TORA"] = value
            if key in gt_combined:
                row_data[f"{key}_TORA_PCT"] = self._calculate_percentage_change(value, gt_combined[key])
        
        # Append to results list
        self.results.append(row_data)
        
        # Update CSV file
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Prepare a row in the correct column order
            row = [row_data.get(col, "") for col in self.columns]
            writer.writerow(row)
        
        # Update plots
        self._update_plots(sample_id)
        
        # Save current plot
        self._save_current_plot(sample_id)
        
        self.logger.info(f"Updated visualization for sample {sample_id}")
    
    def _update_plots(self, sample_id: int) -> None:
        """
        Update the visualization plots with the latest data.
        
        Args:
            sample_id: Current sample ID
        """
        if not self.results:
            return
        
        # Extract data for plotting
        sample_ids = [r.get("sample_id", i) for i, r in enumerate(self.results)]
        
        # Parameters to plot - only showing performance metrics now
        parameters = ["Bandwidth", "PowerConsumption", "VoltageGain"]
        
        # Convert PowerConsumption to mW and Bandwidth to MHz for better visualization
        scale_factors = {
            "PowerConsumption": 1e3,  # Convert from W to mW
            "Bandwidth": 1e-6  # Convert from Hz to MHz
        }
        
        # Ensure axes is always treated as an array
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])
        
        # Update each parameter plot
        for i, param in enumerate(parameters):
            if i >= len(self.axes):
                continue
                
            ax = self.axes[i]
            
            # Clear previous data
            ax.clear()
            
            # Set titles and labels again
            if param == "Bandwidth":
                ax.set_title("Bandwidth")
                ax.set_ylabel("Bandwidth (MHz)")
            elif param == "PowerConsumption":
                ax.set_title("Power Consumption")
                ax.set_ylabel("Power (mW)")
            elif param == "VoltageGain":
                ax.set_title("Voltage Gain")
                ax.set_ylabel("Gain (V/V)")
            
            ax.set_xlabel("Sample ID")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Extract data for this parameter
            gt_data = []
            mlp_data = []
            tora_data = []
            
            for result in self.results:
                # Apply scaling if needed
                scale = scale_factors.get(param, 1.0)
                
                gt_key = f"{param}_GT"
                mlp_key = f"{param}_MLP"
                tora_key = f"{param}_TORA"
                
                if gt_key in result:
                    gt_data.append(result[gt_key] * scale)
                if mlp_key in result:
                    mlp_data.append(result[mlp_key] * scale)
                if tora_key in result:
                    tora_data.append(result[tora_key] * scale)
            
            # Plot the data
            if gt_data:
                ax.plot(sample_ids[:len(gt_data)], gt_data, 'bo-', label='Ground Truth', markersize=4)
            if mlp_data:
                ax.plot(sample_ids[:len(mlp_data)], mlp_data, 'ro-', label='MLP Prediction', markersize=4)
            if tora_data:
                ax.plot(sample_ids[:len(tora_data)], tora_data, 'go-', label='TORA Prediction', markersize=4)
            
            ax.legend()
        
        # Adjust figure size and layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for title
        
        # Update the plot
        plt.ion()
        plt.draw()
        plt.pause(0.01)
    
    def show(self) -> None:
        """Show the plots in a blocking manner."""
        plt.ioff()
        plt.show()
    
    def save_final_results(self) -> None:
        """Save final results to CSV and generate summary plots."""
        # Save the final plot
        final_plot_path = os.path.join(self.output_dir, "final_plot.png")
        self.fig.savefig(final_plot_path, dpi=150)
        
        # Generate a summary DataFrame
        if self.results:
            df = pd.DataFrame(self.results)
            summary_path = os.path.join(self.output_dir, "summary.csv")
            df.to_csv(summary_path, index=False)
            
            # Calculate average percentage changes
            pct_columns = [col for col in df.columns if col.endswith('_PCT')]
            avg_pct = df[pct_columns].mean()
            
            # Save average percentage changes
            avg_path = os.path.join(self.output_dir, "average_pct_changes.csv")
            avg_pct.to_csv(avg_path, header=['Average Percentage Change'])
            
            self.logger.info(f"Saved final results to {self.output_dir}")
        else:
            self.logger.warning("No results to save")

# Test code for the visualizer
if __name__ == "__main__":
    # Simple test with random data
    logging.basicConfig(level=logging.INFO)
    
    output_dir = "visualization_test"
    visualizer = ToraVisualizer(output_dir)
    
    # Generate some test data
    for i in range(10):
        ground_truth = {
            "VDD": 1.5 + np.random.normal(0, 0.1),
            "Vgate": 0.7 + np.random.normal(0, 0.05),
            "Wn": 5e-6 + np.random.normal(0, 1e-6),
            "Rd": 1000 + np.random.normal(0, 100),
            "Bandwidth": 100e6 + np.random.normal(0, 10e6),
            "PowerConsumption": 1e-3 + np.random.normal(0, 1e-4),
            "VoltageGain": 10 + np.random.normal(0, 1)
        }
        
        mlp_prediction = {
            "VDD": ground_truth["VDD"] * (1 + np.random.normal(0, 0.1)),
            "Vgate": ground_truth["Vgate"] * (1 + np.random.normal(0, 0.1)),
            "Wn": ground_truth["Wn"] * (1 + np.random.normal(0, 0.1)),
            "Rd": ground_truth["Rd"] * (1 + np.random.normal(0, 0.1))
        }
        
        mlp_metrics = {
            "Bandwidth": ground_truth["Bandwidth"] * (1 + np.random.normal(0, 0.2)),
            "PowerConsumption": ground_truth["PowerConsumption"] * (1 + np.random.normal(0, 0.2)),
            "VoltageGain": ground_truth["VoltageGain"] * (1 + np.random.normal(0, 0.2))
        }
        
        tora_prediction = {
            "VDD": ground_truth["VDD"] * (1 + np.random.normal(0, 0.05)),
            "Vgate": ground_truth["Vgate"] * (1 + np.random.normal(0, 0.05)),
            "Wn": ground_truth["Wn"] * (1 + np.random.normal(0, 0.05)),
            "Rd": ground_truth["Rd"] * (1 + np.random.normal(0, 0.05))
        }
        
        tora_metrics = {
            "Bandwidth": ground_truth["Bandwidth"] * (1 + np.random.normal(0, 0.1)),
            "PowerConsumption": ground_truth["PowerConsumption"] * (1 + np.random.normal(0, 0.1)),
            "VoltageGain": ground_truth["VoltageGain"] * (1 + np.random.normal(0, 0.1))
        }
        
        visualizer.update(i, ground_truth, mlp_prediction, tora_prediction, mlp_metrics, tora_metrics)
        time.sleep(0.5)  # Simulate processing time
    
    visualizer.save_final_results()
    visualizer.show() 