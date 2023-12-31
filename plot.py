import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

def plot_the_model(trained_weight, trained_bias, feature, label):
  plt.xlabel("feature")
  plt.ylabel("label")

  plt.scatter(feature, label)

  print(trained_bias)
  print(trained_weight)

  x0 = 0
  y0 = trained_bias
  x1 = feature[-1]
  y1 = trained_bias + (trained_weight * x1)
  x_values = [x0, x1]
  y_values = [y0, y1]
  plt.plot(x_values, y_values, c='r')
  plt.show()

def plot_the_loss_curve(epochs, rmse):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()