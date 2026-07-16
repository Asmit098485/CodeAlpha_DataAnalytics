import os
import pandas as pd
import matplotlib.pyplot as plt


# LOAD DATASET


current_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_folder, "titanic.csv")

df = pd.read_csv(csv_path)

print("="*70)
print("EXPLORATORY DATA ANALYSIS (EDA) - TITANIC DATASET")
print("="*70)


# MEANINGFUL QUESTIONS


print("\nQUESTIONS TO EXPLORE")
print("1. How many passengers are there?")
print("2. How many survived?")
print("3. Which passenger class has the most passengers?")
print("4. What is the average age?")
print("5. How many males and females?")
print("6. Are there missing values?")
print("7. Are there duplicate records?")
print("8. What is the fare distribution?")
print("9. Which embarkation port has the most passengers?")


# FIRST & LAST ROWS


print("\nFIRST 5 ROWS")
print(df.head())

print("\nLAST 5 ROWS")
print(df.tail())


# DATA STRUCTURE


print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nDATASET INFORMATION")
df.info()


# MISSING VALUES


print("\nMISSING VALUES")
print(df.isnull().sum())


# DUPLICATES


print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())


# DESCRIPTIVE STATISTICS


print("\nSTATISTICAL SUMMARY")
print(df.describe())


# SURVIVAL


print("\nSURVIVAL COUNT")
print(df["Survived"].value_counts())

print("\nSURVIVAL PERCENTAGE")
print(df["Survived"].value_counts(normalize=True)*100)


# PASSENGER CLASS


print("\nPASSENGER CLASS")
print(df["Pclass"].value_counts())


# GENDER


print("\nGENDER DISTRIBUTION")
print(df["Sex"].value_counts())


# EMBARKED


print("\nEMBARKED PORT")
print(df["Embarked"].value_counts())


# AGE


print("\nAVERAGE AGE")
print(df["Age"].mean())

print("\nMINIMUM AGE")
print(df["Age"].min())

print("\nMAXIMUM AGE")
print(df["Age"].max())


# FARE


print("\nAVERAGE FARE")
print(df["Fare"].mean())

print("\nMAXIMUM FARE")
print(df["Fare"].max())

print("\nMINIMUM FARE")
print(df["Fare"].min())


# CORRELATION


print("\nCORRELATION")
print(df.corr(numeric_only=True))


# GRAPH 1


plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passengers")
plt.show()


# GRAPH 2


plt.figure(figsize=(6,4))
df["Pclass"].value_counts().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Passengers")
plt.show()


# GRAPH 3


plt.figure(figsize=(6,4))
df["Sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Passengers")
plt.show()


# GRAPH 4


plt.figure(figsize=(6,4))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Passengers")
plt.show()

# GRAPH 5


plt.figure(figsize=(6,4))
plt.hist(df["Fare"], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Passengers")
plt.show()


# GRAPH 6


plt.figure(figsize=(6,4))
df["Embarked"].value_counts().plot(kind="bar")
plt.title("Embarked Port Distribution")
plt.xlabel("Port")
plt.ylabel("Passengers")
plt.show()


# CONCLUSION


print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print("""
1. The dataset contains passenger information from the Titanic.
2. The dataset has both numerical and categorical variables.
3. Missing values are present in Age, Cabin and Embarked.
4. There are very few or no duplicate records.
5. Most passengers did not survive.
6. Third-class passengers are the highest in number.
7. Male passengers are more than female passengers.
8. Fare values vary significantly.
9. The dataset is suitable for further Machine Learning analysis.
""")

print("\nEDA COMPLETED SUCCESSFULLY")