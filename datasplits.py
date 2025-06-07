import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\894054\Downloads\maindataset.csv")

# Perform 3 different 70/30 splits
for i in range(1, 4):
    train, test = train_test_split(df, test_size=0.3, random_state=i)
    train.to_csv(f"tra1n_split_{i}.csv", index=False)
    test.to_csv(f"t3st_split_{i}.csv", index=False)
    print(f"Split {i}: Train size = {len(train)}, Test size = {len(test)}")
