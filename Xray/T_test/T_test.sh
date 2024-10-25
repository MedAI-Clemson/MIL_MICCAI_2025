#!/bin/bash

# Function to run the Python script for a pair of models
compare_models() {
    local model1=$1
    local model2=$2
    
    echo "Comparing ${model1} and ${model2}..."
    python T_test.py ${model1} ${model2}
    echo "Comparison between ${model1} and ${model2} completed."
    echo "----------------------------------------"
}

# Main execution
echo "Starting MIL model comparisons..."

# Run comparisons for different model pairs
compare_models "MIL1" "MIL2"
compare_models "MIL2" "MIL3"
compare_models "MIL3" "MIL4"

echo "All comparisons completed."