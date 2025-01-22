import logging
from pathlib import Path
import os
from ModelOne.modelone import load_model, predict_single_file  # Import necessary functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Path to the saved model
    model_path = Path('../ModelOne/model.joblib')  # Use Path for model path

    # Debug: Print the absolute path of the model file
    logging.info(f"Looking for model file at: {model_path.absolute()}")

    # Check if the model file exists
    if not model_path.exists():
        logging.error(f"Model file not found at {model_path}. Please train the model first.")
    else:
        # Load the model and preprocessing objects
        model, pca, imputer, masker = load_model(model_path)
        logging.info("Model and preprocessing objects loaded successfully.")

        # Construct the NIfTI file path using an absolute path or correct relative path
        # Option 1: Use an absolute path (recommended for debugging)
        nii_file_path = Path('/Users/mennahtullahmabrouk/PycharmProjects/graduationproject/ModelOne/abide/ABIDE_pcp/cpac/nofilt_noglobal/Caltech_0051461_func_preproc.nii.gz')

        # Option 2: Use a relative path from the project root (if the script is run from the project root)
        # nii_file_path = Path('ModelOne/abide/ABIDE_pcp/cpac/nofilt_noglobal/Caltech_0051461_func_preproc.nii.gz')

        # Debug: Print the absolute path of the NIfTI file
        logging.info(f"Looking for NIfTI file at: {nii_file_path.absolute()}")

        # Debug: Check if the file exists using os.path.exists
        logging.info(f"File exists: {os.path.exists(nii_file_path)}")

        if not nii_file_path.exists():
            logging.error(f"NIfTI file not found at {nii_file_path}.")
        else:
            # Make a prediction
            result = predict_single_file(nii_file_path, model, pca, imputer, masker)
            print(result)