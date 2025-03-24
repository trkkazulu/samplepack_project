import os
import subprocess  # To run external commands
import time

def run_script(script_name):
    """
    Runs a Python script using subprocess.
    """
    print(f"\n--- Running {script_name} ---")
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stderr)
        raise  # Re-raise the exception to stop the sequence

def main():
    """
    Orchestrates the execution of the AI bass sample pack generation scripts
    in the correct sequence.
    """
    try:
        # 1. Generate Metadata
        run_script("generate_metadata.py")

        # 2. Fix Names (and update metadata)
        run_script("fix_names.py")

        # 3. Data Preparation
        run_script("data_preparation.py")

        # 4. Model Creation (Assumed to be part of training.py or defined in model.py)
        # Note: Model creation is often integrated into the training script or defined
        #       in a separate model.py file. If you have a separate model.py that
        #       needs to be run, uncomment the following line:
        # run_script("model.py")

        # 5. Model Training
        run_script("training.py")

        # 6. Prediction
        run_script("prediction.py")

        print("\n--- AI Bass Sample Pack Generation Process Complete! ---")

    except Exception as e:
        print(f"\n--- Process Aborted due to error: {e} ---")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal time taken: {total_time:.2f} seconds")
