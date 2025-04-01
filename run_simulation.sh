#!/bin/bash

# Load Cadence module
module load cadence/virtuoso/6.18

# Export the path to ensure ocean is in the PATH
export PATH="/EDA_Tools/Cadence/IC618/tools/dfII/bin:$PATH"

# Run the simulation script with provided arguments
/home/divake/miniconda3/envs/env_cu121/bin/python simulation.py "$@"

# Check the exit status
if [ $? -ne 0 ]; then
    echo "Simulation failed."
    exit 1
fi

echo "Simulation completed successfully." 