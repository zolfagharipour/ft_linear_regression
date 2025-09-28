# ft_linear_regression

An introduction to machine learning - Linear regression implementation with gradient descent algorithm.

## Description

This project implements a simple linear regression algorithm to predict car prices based on mileage. The implementation includes:

- **Training program**: Uses gradient descent to learn the optimal parameters (θ₀, θ₁) for the linear function `price = θ₀ + θ₁ × mileage`
- **Prediction program**: Interactive tool to predict car prices for given mileage values
- **Evaluation program**: Calculates model performance metrics (MSE, RMSE, MAE)
- **Visualization**: Generates training plots and animated GIF showing the learning process

The algorithm uses feature standardization and implements the mathematical formulas:
- **Cost function**: J(θ) = (1/(2m)) × Σ(θ₀ + θ₁×xᵢ - yᵢ)²
- **Gradient descent**: θ₀ and θ₁ are updated simultaneously using partial derivatives

## Warning for 42 Students

This repository is intended as a reference and educational tool. **42 students are strongly advised not to copy this code without fully understanding its functionality.** Plagiarism in any form is against 42's principles and could lead to serious academic consequences. Use this repository responsibly to learn and better understand how to implement similar functionalities on your own.

## Features

- ✅ **Gradient Descent Implementation**: Custom implementation without using libraries like `numpy.polyfit`
- ✅ **Feature Standardization**: Proper data preprocessing for better convergence
- ✅ **Interactive Prediction**: Command-line interface for price predictions
- ✅ **Model Persistence**: Saves trained parameters to JSON file
- ✅ **Performance Evaluation**: MSE, RMSE, and MAE calculations
- ✅ **Visualization**: Training progress plots and animated GIF
- ✅ **Error Handling**: Robust input validation and file operations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ft_linear_regression
```

2. Set up virtual environment and install dependencies:
```bash
make setup
```

## Usage

### Training the Model
```bash
make train
# or directly:
python src/train.py data/data.csv
```

This will:
- Load the dataset from `data/data.csv`
- Train the model using gradient descent (10 epochs by default)
- Save the learned parameters to `model/model.json`
- Generate training plots in `plots/` directory
- Create an animated GIF showing the learning process

### Making Predictions
```bash
make predict
# or directly:
python src/predict.py
```

Interactive session to predict car prices:
```
Enter mileage (or 'q' to quit): 150000
Estimated price: 4500.00
```

### Evaluating Model Performance
```bash
make evaluate
# or directly:
python src/evaluate.py
```

Outputs performance metrics:
```
Mean Squared Error (MSE): 1234.56
Root Mean Squared Error (RMSE): 35.14
Mean Absolute Error (MAE): 28.90
```

## Project Structure

```
ft_linear_regression/
├── data/
│   └── data.csv          # Training dataset (mileage, price)
├── model/
│   └── model.json        # Saved model parameters
├── plots/                # Training visualization plots
├── src/
│   ├── train.py          # Training program
│   ├── predict.py        # Prediction program
│   └── evaluate.py       # Evaluation program
├── Makefile              # Build automation
├── requirements.txt      # Python dependencies
└── README.md
```


## Dependencies

- Python 3.x
- NumPy 2.1.1
- Matplotlib 3.10
- Pillow 10.0.0


## License

This project is for educational purposes as part of the 42 curriculum.
