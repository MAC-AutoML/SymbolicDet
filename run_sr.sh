#!/bin/bash

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Default parameters
ENABLE_LLM=True
CONFIG_PATH="config/default_config.yaml"
DEFAULT_CONFIG_PATH="config/default_config.yaml"

API_KEY=${SR_API_KEY:-""}  # Get API_KEY from environment variable, default to empty if not set

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                  Show help information"
    echo "  --no-llm                    Disable LLM feature"
    echo "  -c, --config <path>         Specify configuration file path"
    echo "  -k, --api-key <key>         Set API KEY (can also be set through SR_API_KEY environment variable)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-llm)
            ENABLE_LLM=False
            shift
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if LLM is enabled and API_KEY is set
if [ "$ENABLE_LLM" = true ] && [ -z "$API_KEY" ]; then
    echo "Error: API KEY is required when enabling LLM"
    echo "You can set it in the following ways:"
    echo "1. Environment variable: export SR_API_KEY=your_api_key"
    echo "2. Command line argument: -k your_api_key"
    exit 1
fi

# Check if configuration file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "No configuration file specified, using default configuration: $DEFAULT_CONFIG_PATH"
    CONFIG_PATH="$DEFAULT_CONFIG_PATH"
fi

# Create output directory
mkdir -p output

# If API_KEY is provided through command line, set environment variable
if [ -n "$API_KEY" ]; then
    export SR_API_KEY="$API_KEY"
fi

# Run Python script
echo "Starting symbolic regression generation..."
echo "Configuration file: $CONFIG_PATH"
echo "LLM feature: $ENABLE_LLM"

python src/run_sr.py \
    --enable_llm=$ENABLE_LLM \
    --config_path="$CONFIG_PATH"

# Check run result
if [ $? -eq 0 ]; then
    echo "Run completed"
else
    echo "Run failed"
    exit 1
fi 