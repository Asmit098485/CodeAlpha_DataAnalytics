import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load Dataset


current_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_folder, "titanic.csv")

df = pd.read_csv(csv_path)


# Display Basic Information


print("=" * 70)
print("CODEALPHA TASK 3 : DATA VISUALIZATION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


# Graph 1 : Survival Count


plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()


# Graph 2 : Passenger Class


plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()


# Graph 3 : Gender Distribution


plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()


# Graph 4 : Age Distribution


plt.figure(figsize=(7,4))
sns.histplot(df["Age"].dropna(), bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Graph 5 : Fare Distribution


plt.figure(figsize=(7,4))
sns.histplot(df["Fare"], bins=20, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Graph 6 : Embarked Port Distribution


plt.figure(figsize=(6,4))
sns.countplot(x="Embarked", data=df)
plt.title("Embarked Port Distribution")
plt.xlabel("Port")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()


# Graph 7 : Correlation Heatmap


plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Graph 8 : Fare Boxplot


plt.figure(figsize=(6,4))
sns.boxplot(y=df["Fare"])
plt.title("Fare Boxplot (Outlier Detection)")
plt.tight_layout()
plt.show()


# Graph 9 : Survival by Gender


plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.tight_layout()
plt.show()


# Graph 10 : Survival by Passenger Class


plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.show()


# Conclusion


print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Insights:

1. Most passengers did not survive.
2. Third-class passengers were the largest group.
3. Male passengers were more than female passengers.
4. Most passengers were between 20–40 years old.
5. Ticket fares varied significantly.
6. Southampton (S) had the highest number of passengers.
7. Fare contains several outliers.
8. Female passengers had a higher survival rate.
9. First-class passengers survived more often than third-class passengers.
""")