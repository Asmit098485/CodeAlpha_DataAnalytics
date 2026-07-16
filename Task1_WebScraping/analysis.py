import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("books.csv")

# Remove £ symbol and convert to float
df["Price"] = df["Price"].str.replace("Â£", "", regex=False)
df["Price"] = df["Price"].str.replace("£", "", regex=False)
df["Price"] = df["Price"].astype(float)

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Basic statistics
print("\nAverage Price:", df["Price"].mean())
print("Highest Price:", df["Price"].max())
print("Lowest Price:", df["Price"].min())

# Histogram
plt.hist(df["Price"], bins=10)
plt.title("Distribution of Book Prices")
plt.xlabel("Price")
plt.ylabel("Number of Books")
plt.show()