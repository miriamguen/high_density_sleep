import os
import sys
import subprocess
import logging
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)


def run_script(script_name, description):
    """Run a script and handle any errors that occur."""
    logging.info(f"Starting {description}...")
    try:
        result = subprocess.run(
            [sys.executable, script_name], check=True, capture_output=True, text=True
        )
        logging.info(f"Completed {description} successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in {description}:")
        logging.error(f"Exit code: {e.returncode}")
        logging.error(f"Output: {e.output}")
        logging.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error in {description}: {str(e)}")
        return False


def main():
    # Load parameters
    with open("analysis_code/parameters.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Define the pipeline steps
    pipeline = [
        {
            "script": "analysis_code/preprocessing_main.py",
            "description": "Data preprocessing",
        },
        {
            "script": "analysis_code/decomposition_main.py",
            "description": "Feature decomposition",
        },
        {
            "script": "analysis_code/supervised_scoring.py",
            "description": "Supervised scoring",
        },
        {
            "script": "analysis_code/component_statistics.py",
            "description": "statistical modeling of components",
        },
        {
            "script": "analysis_code/hmm_main.py",
            "description": "run the hmm analysis",
        },
        {
            "script": "analysis_code/plot_state_map.py",
            "description": "plot the feature maps representing the hidden states",
        },
        {
            "script": "analysis_code/visualize_most_typical.py",
            "description": "plot the eeg examples best representing the hidden states for each subject",
        },
    ]

    # Create output directory if it doesn't exist
    output_dir = Path(params["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save a copy of the parameters file in the output directory
    with open(output_dir / "parameters.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False)

    # Run each step in the pipeline
    for step in pipeline:
        if not run_script(step["script"], step["description"]):
            logging.error(f"Pipeline failed at {step['description']}")
            sys.exit(1)

    logging.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
