# ft_linear_regression

An introduction to machine learning by implementing a linear regression model from scratch to predict car prices based on mileage.

## 🛠️ Installation

1. **Python 3.7+** required.
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

## 🚀 Usage

1. Train the Model
	```bash
	python training.py
	```
	- Reads `data/data.csv`.
	- Outputs:
		- `data/model.txt`: Trained parameters (θ₀, θ₁).
		- `graphs/cost_plot.png`: Cost convergence over iterations.
		- `graphs/price_vs_km_plot.png`: Data distribution and regression line.
2. Predict a Price
	```bash
	python prediction.py
	```
3. Evaluate a model performance
	```bash
	python evaluation.py
	```
## 📊 Visualization

- **Data Distribution & Regression Line** (`price_vs_km_plot.png`):  
	A scatter plot showing the relationship between car mileage (km) and price, with the regression line representing the model's predictions.
- **Cost Convergence** (`cost_plot.png`):
	Tracks how the cost decreases over training iterations.

## 📜 License

MIT License. Free to use and modify. See [LICENSE](https://spdx.org/licenses/MIT.html) for details.