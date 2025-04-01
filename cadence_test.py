#!/usr/bin/env python3
import subprocess
import os
import sys
import tempfile
import time

def run_command(cmd, verbose=True):
    """Run a command and return its output"""
    if verbose:
        print(f"Running command: {cmd}")
    
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    
    if verbose:
        if stdout:
            print(f"Output:\n{stdout}")
        if stderr:
            print(f"Error output:\n{stderr}")
    
    return stdout, stderr, process.returncode

def check_module_availability():
    """Check if the Cadence modules are available"""
    print("\n=== Checking Module Availability ===")
    stdout, stderr, return_code = run_command("module avail 2>&1 | grep -i cadence")
    
    if return_code != 0:
        print("No Cadence modules found. Make sure modules are properly set up.")
        return False
    
    return True

def load_cadence_module():
    """Load the Cadence Virtuoso module"""
    print("\n=== Loading Cadence Module ===")
    stdout, stderr, return_code = run_command("module load cadence/virtuoso/6.18")
    
    if return_code != 0:
        print("Failed to load Cadence module. Check module availability and permissions.")
        return False
    
    return True

def check_virtuoso_availability():
    """Check if Virtuoso is available in PATH"""
    print("\n=== Checking Virtuoso Availability ===")
    stdout, stderr, return_code = run_command("which virtuoso")
    
    if return_code != 0:
        print("Virtuoso not found in PATH. Check if Cadence module is loaded.")
        return False
    
    stdout, stderr, return_code = run_command("virtuoso -version")
    return return_code == 0

def check_spectre_availability():
    """Check if Spectre is available in PATH"""
    print("\n=== Checking Spectre Availability ===")
    stdout, stderr, return_code = run_command("which spectre")
    
    if return_code != 0:
        print("Spectre not found in PATH. Check if Cadence module is loaded.")
        return False
    
    stdout, stderr, return_code = run_command("spectre -version")
    return return_code == 0

def check_license():
    """Check if license is properly set"""
    print("\n=== Checking License Configuration ===")
    stdout, stderr, return_code = run_command("echo $LM_LICENSE_FILE")
    
    if '5280@cadenceuic.webstore.illinois.edu' not in stdout:
        print("License server not properly configured.")
        return False
    
    return True

def create_simple_netlist():
    """Create a simple netlist for testing"""
    print("\n=== Creating Simple Test Netlist ===")
    
    # Create a temporary directory
    test_dir = tempfile.mkdtemp(prefix="cadence_test_")
    print(f"Created test directory: {test_dir}")
    
    # Create a simple netlist file
    netlist_path = os.path.join(test_dir, "simple_circuit.scs")
    
    with open(netlist_path, 'w') as f:
        f.write("""// Simple RC circuit for Spectre testing
simulator lang=spectre

// Power supply
vdd (in 0) vsource dc=1.8 type=pulse val0=0 val1=1.8 period=20n rise=1n fall=1n width=10n

// Simple RC circuit
r1 (in out) resistor r=1k
c1 (out 0) capacitor c=1p

// Analysis statements
tran tran step=0.01n stop=50n
""")
    
    return test_dir, netlist_path

def run_simple_simulation(netlist_path):
    """Run a simple Spectre simulation"""
    print(f"\n=== Running Simple Simulation with Netlist: {netlist_path} ===")
    
    cmd = f"cd {os.path.dirname(netlist_path)} && spectre {os.path.basename(netlist_path)} -format psfascii"
    stdout, stderr, return_code = run_command(cmd)
    
    if return_code != 0:
        print("Simulation failed. Check spectre error output.")
        return False
    
    # Check if simulation output was created
    result_dir = os.path.join(os.path.dirname(netlist_path), "simple_circuit.raw")
    if not os.path.exists(result_dir):
        print(f"Simulation output directory not found: {result_dir}")
        return False
    
    print(f"Simulation completed successfully. Results in: {result_dir}")
    return True

def cleanup(test_dir):
    """Clean up temporary files"""
    print(f"\n=== Cleaning up test directory: {test_dir} ===")
    
    import shutil
    try:
        shutil.rmtree(test_dir)
        print("Cleanup completed.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def main():
    """Main function to run all checks"""
    print("=== Cadence Virtuoso & Spectre Test Script ===")
    
    # Check module availability
    if not check_module_availability():
        return 1
    
    # Load Cadence module
    if not load_cadence_module():
        return 1
    
    # Check Virtuoso availability
    if not check_virtuoso_availability():
        return 1
    
    # Check Spectre availability
    if not check_spectre_availability():
        return 1
    
    # Check license configuration
    if not check_license():
        return 1
    
    # Create a simple netlist for testing
    test_dir, netlist_path = create_simple_netlist()
    
    # Run a simple simulation
    simulation_success = run_simple_simulation(netlist_path)
    
    # Clean up
    cleanup(test_dir)
    
    if simulation_success:
        print("\n=== All tests completed successfully! ===")
        return 0
    else:
        print("\n=== Some tests failed. Please check the output above. ===")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 