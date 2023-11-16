import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from sklearn.linear_model import LinearRegression

def view(position=None, scale=None):
  # 非線形関数
  def function_f(x):
      return - x - 0.15 * np.sin(20*x) + 1.15
  
  # Define a non-linear distance scale
  def distance_scale(x):
      return np.abs(x)

  T = [0.77]
  samples_x = skewnorm.rvs(position, loc=T, scale=scale, size=100)
  weights = np.exp(-((T - samples_x - position*0.05)**2))

  # Calculate the function values for samples
  samples_f = function_f(samples_x)

  # Prepare data for weighted linear regression
  X = samples_x.reshape(-1, 1)
  y = samples_f.reshape(-1, 1)
  w = weights.reshape(-1, 1)

  # Perform weighted linear regression
  model = LinearRegression()
  model.fit(X, y, sample_weight=w.ravel())

  # Predict values for a range of x values
  x_range = np.linspace(0, 1, 100).reshape(-1, 1)
  predictions = model.predict(x_range)

  # Plotting
  fig, ax1 = plt.subplots(figsize=(10, 6))

  # Plot the histogram of the samples in the background with secondary y-axis
  ax2 = ax1.twinx()
  # Use the same x-range for the histogram bins as for the x-axis.
  hist_bins = np.linspace(0, 1, 30)
  hist = ax2.hist(samples_x, bins=hist_bins, color='grey', alpha=0.5, label='Samples Histogram')
  ax2.set_ylim(0, 20)  # Set the limit of y-axis for histogram to 200
  ax2.set_ylabel('Number of Samples in Bin')

  # Plot the nonlinear function f
  x_vals = np.linspace(0, 1, 100)
  y_vals = function_f(x_vals)
  ax1.plot(x_vals, y_vals, label='Nonlinear Function f', color='black')

  # Plot the samples with weighted markers
  for x, y, weight in zip(samples_x, samples_f, weights):
      ax1.scatter(x, y, s=weight * 1000, c='red' if y <= 0.5 else 'blue', marker='+' if y <= 0.5 else 'o')

  ax1.scatter(T, function_f(T[0]), s=1000, c='black', marker='+')

  # Plot the linear model
  ax1.plot(x_range, predictions, label='Weighted Linear Regression Model', color='black', linestyle='-')

  # Customize the plot
  ax1.set_xlabel('X-axis')
  ax1.set_ylabel('Y-axis')
  ax1.set_xlim(0.2, 1)
  ax1.set_ylim(0, 1)
  ax1.set_title('Plot of Nonlinear Function with Weighted Samples, Histogram, and Linear Regression Model')
  ax1.legend(loc='upper left')
  ax2.legend(loc='upper right')
  plt.show()
  plt.savefig(f'05Discussion/Position{position}_Scale{scale}.png')


# Usage example
view(position=-8, scale=0.3)
view(position=0.5, scale=0.1)
