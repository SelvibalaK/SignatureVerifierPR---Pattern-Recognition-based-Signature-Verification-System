# SignatureVerifierPR---Pattern-Recognition-based-Signature-Verification-System
## Overview
This project implements a pattern recognition-based signature verification system to detect whether a given signature is genuine or forged. The system leverages convolutional neural networks (CNNs) for classifying signatures, utilizing two types of image datasets: real signatures and forged signatures.

## Dataset
The dataset consists of two categories of images:
- Real Signatures: Authentic signatures that serve as the reference for comparison.
- Forged Signatures: Fake or forged signatures used to train the model on fraudulent patterns.
The images are preprocessed to ensure uniformity in size and quality, allowing the model to accurately distinguish between real and forged signatures.

## Features
- **CNN-based pattern recognition:** The system uses a CNN model to extract signature features and classify them.
- **Data augmentation:** Includes rotation, shear, and zoom augmentation techniques to improve model generalization.
- **Prediction of Forged or Genuine:** Once trained, the model can predict whether a new signature is real or forged.

## Installation
### Prerequisites
Ensure you have the following libraries installed:
 pip install tensorflow opencv-python scikit-learn numpy

### Instructions
1. Clone the repository
2. Prepare the dataset
Download or organize your dataset with two folders:
real/: Real signatures.
forge/: Forged signatures.
3. Adjust the paths in the script to point to your dataset.

4. Run the Python script in your environment (e.g., Jupyter Notebook).

## Running the Project
1. Preprocess the data and train the CNN model on the signature dataset.
2. After training, the model will be able to evaluate new signature images for their authenticity.
The system will return either "Genuine" or "Forged" based on the input image.

## Development
This project is fully developed as a standalone signature verification system. Contributions or suggestions for improvement are welcome.

## License
This project is licensed under the MIT License.
