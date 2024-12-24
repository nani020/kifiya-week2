import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data["Date"], data["Close"], label="Close Price", color="blue")
    plt.plot(data["Date"], data["Moving Average"], label="Moving Average", color="red")
    plt.xlabel("Data")
    plt.ylabel("Price")
    plt.title("Stock price, moving average")
    plt.legend()
    plt.grid()
    plt.show()

def plot_univariate(data, column, title=None):
    """
    Function to plot a univariate analysis (distribution of a single variable).
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(title or f"Univariate Analysis of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_bivariate(data, x_column, y_column, title=None):
    """
    Function to plot a bivariate analysis (scatter plot between two variables).
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[x_column], y=data[y_column])
    plt.title(title or f"Bivariate Analysis: {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()