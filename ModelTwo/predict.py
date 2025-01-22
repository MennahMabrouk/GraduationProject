import os
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path  # Import pathlib for robust path handling

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the model architecture
class MTAutoEncoder(nn.Module):
    def __init__(self, num_inputs=6670, num_latent=3335, tied=True, use_dropout=False):
        super(MTAutoEncoder, self).__init__()
        self.tied = tied
        self.num_latent = num_latent
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(num_latent, 1)
        )

    def forward(self, x, eval_classifier=False):
        x = torch.tanh(self.fc_encoder(x))
        x_logit = self.classifier(x) if eval_classifier else None
        x_rec = F.linear(x, self.fc_encoder.weight.t()) if self.tied else self.fc_decoder(x)
        return x_rec, x_logit

def preprocess_single_file(filepath):
    logger.info(f"Processing file: {filepath}")
    # Load the .1D file
    try:
        data = np.loadtxt(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None

    if data.ndim == 1:  # If the data is 1D, reshape it to (timepoints, 1)
        data = data.reshape(-1, 1)
    elif data.ndim == 2:  # If the data is 2D, ensure it's (timepoints, regions)
        if data.shape[0] < 2:  # Check if there are at least 2 timepoints
            logger.warning(f"Skipping {filepath}: Not enough timepoints for correlation matrix calculation.")
            return None
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # Compute the correlation matrix
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(data.T))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        flattened = ma.masked_where(m, corr).compressed()

    # Truncate or pad the flattened vector to match the model's input size
    target_size = 3334  # Match the model's input size
    if len(flattened) > target_size:
        flattened = flattened[:target_size]  # Truncate
    elif len(flattened) < target_size:
        flattened = np.pad(flattened, (0, target_size - len(flattened)))  # Pad with zeros

    return flattened

# Load the pre-trained model
def load_model(model_path, num_inputs=3334, num_latent=1667):
    model = MTAutoEncoder(num_inputs=num_inputs, num_latent=num_latent, tied=True, use_dropout=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Make predictions
def predict(model, new_data):
    with torch.no_grad():
        _, logits = model(new_data.to(device), eval_classifier=True)
        proba = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities
        prediction = 1 if proba >= 0.5 else 0  # Binary classification threshold
    return prediction, proba

# Main function for prediction
def main(filepath):
    # Convert filepath to a Path object
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return

    # Define paths
    base_dir = Path(__file__).parent
    model_path = base_dir / 'model.pth'  # Path to the trained model

    # Check if the model file exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    # Define model parameters (must match the ones used during training)
    num_inputs = 3334  # Update to match the pre-trained model
    num_latent = 1667  # Update to match the pre-trained model

    # Load the model
    model = load_model(model_path, num_inputs, num_latent)

    # Preprocess the single file
    preprocessed_data = preprocess_single_file(filepath)
    if preprocessed_data is None:
        logger.error("Invalid data. Exiting.")
        return

    # Convert to tensor and add batch dimension
    preprocessed_data = torch.FloatTensor(preprocessed_data).unsqueeze(0)

    # Make predictions
    prediction, probability = predict(model, preprocessed_data)

    # Output results
    logger.info(f"Prediction: {'ASD' if prediction == 1 else 'Non-ASD'}")
    logger.info(f"Probability: {probability[0][0]:.4f}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if __name__ == "__main__":
    import argparse
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict ASD from a single .1D file.")
    parser.add_argument("filepath", type=str, help="Path to the .1D file for prediction.")
    args = parser.parse_args()

    # Run prediction
    main(args.filepath)