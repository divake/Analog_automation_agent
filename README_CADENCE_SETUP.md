# Cadence EDA Tools Setup Guide

This document provides comprehensive information about the Cadence EDA tools setup in our environment. Use this as a reference for future work with Cadence tools.

## Overview

Cadence Design Systems provides Electronic Design Automation (EDA) tools for integrated circuit (IC) design, including:

- **Virtuoso**: Layout and schematic editor for IC design
- **Spectre**: Circuit simulator for analog and mixed-signal simulation
- **ADE**: Analog Design Environment for simulation setup and analysis
- **Assura/Pegasus**: Physical verification tools

## Environment Setup

### Module System

Cadence tools are available through the module system:

```bash
# List available Cadence modules
module avail | grep -i cadence

# Load Virtuoso 6.18 (includes Spectre)
module load cadence/virtuoso/6.18
```

### Directory Structure

Cadence tools are installed in the `/EDA_Tools` directory:

```
/EDA_Tools/
├── Cadence/
│   ├── Virtuoso/
│   │   └── 6.18/
│   ├── Spectre/
│   │   └── 23.1.0/
│   └── ...
└── ...
```

### Version Information

- **Virtuoso**: 6.1.8-64b
- **Spectre**: 23.1.0

## License Configuration

The license server is configured at:
```
5280@cadenceuic.webstore.illinois.edu
```

The license is accessible through the environment variable:
```bash
echo $LM_LICENSE_FILE
```

## Basic Usage

### Starting Virtuoso

After loading the module, start Virtuoso with:

```bash
virtuoso &
```

### Running Spectre Simulations from Command Line

Run a Spectre simulation with:

```bash
spectre <netlist_file>.scs -format psfascii
```

### Simple Netlist Example

Here's a basic netlist for a simple RC circuit:

```
// Simple RC circuit for Spectre testing
simulator lang=spectre

// Power supply
vdd (in 0) vsource dc=1.8 type=pulse val0=0 val1=1.8 period=20n rise=1n fall=1n width=10n

// Simple RC circuit
r1 (in out) resistor r=1k
c1 (out 0) capacitor c=1p

// Analysis statements
tran tran step=0.01n stop=50n
```

Save this to a `.scs` file and run with Spectre.

## Verification Script

The repository includes `cadence_test.py` which performs the following checks:

1. Verifies module availability
2. Loads the Cadence module
3. Checks Virtuoso availability
4. Checks Spectre availability
5. Verifies license configuration
6. Creates a simple netlist and runs a test simulation

Run this script to verify your environment:

```bash
python3 cadence_test.py
```

## Common Issues and Troubleshooting

### License Issues

If you encounter license errors:

1. Verify that the license server is running:
   ```bash
   echo $LM_LICENSE_FILE
   ```

2. Check connectivity to the license server:
   ```bash
   ping cadenceuic.webstore.illinois.edu
   ```

3. Request license status:
   ```bash
   lmstat -c $LM_LICENSE_FILE -a
   ```

### Path Issues

If tools aren't found after loading the module:

1. Verify the module loaded correctly:
   ```bash
   module list
   ```

2. Check if the tools are in your PATH:
   ```bash
   which virtuoso
   which spectre
   ```

3. Manually add to PATH if needed:
   ```bash
   export PATH=/EDA_Tools/Cadence/Virtuoso/6.18/bin:$PATH
   ```

### Simulation Errors

For netlist and simulation errors:

1. Verify syntax with simpler circuits first
2. Check model file paths in your netlist
3. Review Spectre manual for correct component parameters

## Additional Resources

- Internal documentation: `/EDA_Tools/Cadence/docs/`
- Official Cadence Documentation: Access through Virtuoso Help menu
- University Support: Contact lab administrator for specific setup issues

## Setting Up New Projects

1. Create a new directory for your project
2. Start Virtuoso after loading the module
3. Create a new library in Virtuoso
4. Set up technology files as needed for your process
5. Begin creating schematics, running simulations, and designing layouts

## Notes for AICircuit Project

For the AICircuit project:
1. Always load the cadence/virtuoso/6.18 module before starting work
2. Ensure VPN connection is active when accessing tools remotely
3. Store designs in a version-controlled repository
4. Run the verification script periodically to ensure environment integrity 