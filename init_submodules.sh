#!/bin/bash

git submodule update --init --recursive

# Initialize variables to store path and branch
current_path=""
current_branch=""

# Read the file line by line
while IFS= read -r line; do
    # Check if the line contains a path
    if [[ $line =~ ^[[:space:]]*path[[:space:]]*=[[:space:]]*(.*) ]]; then
        current_path="${BASH_REMATCH[1]}"
    fi
    
    # Check if the line contains a branch
    if [[ $line =~ ^[[:space:]]*branch[[:space:]]*=[[:space:]]*(.*) ]]; then
        current_branch="${BASH_REMATCH[1]}"
        
        # Print the path and branch once both are found
        echo "Path: $current_path"
        cd $current_path
        echo "Branch: $current_branch"
        git checkout $current_branch
        git pull
        cd ..
        echo ""
    fi
done < .gitmodules