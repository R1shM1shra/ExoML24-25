import pandas as pd


file_path = r"C:\Users\894054\Downloads\cumulative_2025.01.28_13.40.32.csv"
data = pd.read_csv(file_path)

# Check for null values before processing
print("Null values per column before processing:\n", data.isnull().sum())

# Remove rows with any null values
data_cleaned = data.dropna()

# Display dataset info after removing null values
print("\nDataset after removing null values:")
print(data_cleaned.info())

# Save the cleaned dataset to a new CSV file
cleaned_file_path = "cleaned_dataset.csv"
data_cleaned.to_csv(cleaned_file_path, index=False)

print(f"\nCleaned dataset saved as '{cleaned_file_path}'")