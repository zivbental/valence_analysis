import pandas as pd

# Load CSV file into a pandas DataFrame
df = pd.read_csv(r"C:\temp_data\Valence\ga_post_oct\15.12.2025\trial_2\fly_loc.csv")

# Create a dataframe with only experiment_step and chamber_x_loc columns (x = 1 to 20)
columns_to_select = ['experiment_step'] + [f'chamber_{i}_loc' for i in range(1, 21)]
df = df[columns_to_select]

# Split the dataframe to two dataframes
odor_right_df = df[df['experiment_step'] == 'Odor Right']
odor_left_df = df[df['experiment_step'] == 'Odor Left']

# For each column, perform the calculation
chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
results = []
for col in chamber_columns:
    # Calculate ratio for odor_right_df: number of rows where value > 0 / total rows
    odor_right_ratio = (odor_right_df[col] < 0).sum() / len(odor_right_df)
    # Calculate ratio for odor_left_df: number of rows where value > 0 / total rows
    odor_left_ratio = (odor_left_df[col] < 0).sum() / len(odor_left_df)
    # Subtract odor_left from odor_right and multiply by 100
    result = (odor_right_ratio - odor_left_ratio) * 100
    results.append(result)

# Create result dataframe
result_df = pd.DataFrame({
    'chamber': [f'chamber_{i}' for i in range(1, 21)],
    'result': results
})

# Export result df to output.csv
result_df.to_csv('output.csv', index=False)

# Calculate and print statistical properties
import numpy as np
average = result_df['result'].mean()
sem = result_df['result'].std() / np.sqrt(len(result_df))
print(f"\nAverage: {average}")
print(f"SEM: {sem}")

# Display the DataFrame
print(df.head())
print(f"\nDataFrame shape: {df.shape}")
