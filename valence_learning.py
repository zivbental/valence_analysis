import pandas as pd
import numpy as np

# Load CSV file into a pandas DataFrame
df = pd.read_csv("fly_loc.csv")

# Create a dataframe with only experiment_step and chamber_x_loc columns (x = 1 to 20)
columns_to_select = ['experiment_step'] + [f'chamber_{i}_loc' for i in range(1, 21)]
df = df[columns_to_select]


def filter_moving_flies(df, threshold_X=60, min_crossings_Y=10):
    """
    Filter out flies that haven't moved enough.
    
    Parameters:
    - threshold_X: Position threshold (e.g., 50 means positions 50 and -50)
    - min_crossings_Y: Minimum number of times to cross each threshold
    
    Returns:
    - List of chamber column names to keep
    """
    chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
    valid_chambers = []
    
    for col in chamber_columns:
        positions = df[col].dropna().values
        
        if len(positions) < 2:
            continue  # Skip if not enough data
        
        # Count crossings of +X (going from < X to >= X)
        crossings_plus_X = 0
        for i in range(len(positions) - 1):
            if positions[i] < threshold_X and positions[i+1] >= threshold_X:
                crossings_plus_X += 1
        
        # Count crossings of -X (going from > -X to <= -X)
        crossings_minus_X = 0
        for i in range(len(positions) - 1):
            if positions[i] > -threshold_X and positions[i+1] <= -threshold_X:
                crossings_minus_X += 1
        
        # Check if crossing counts meet threshold
        meets_plus_threshold = crossings_plus_X >= min_crossings_Y
        meets_minus_threshold = crossings_minus_X >= min_crossings_Y
        
        # Keep if either crossing count meets threshold (passed +X OR -X at least Y times)
        if meets_plus_threshold or meets_minus_threshold:
            valid_chambers.append(col)
    
    return valid_chambers


# Split the dataframe into four experiment steps
odor_right_df_before_shock = df[df['experiment_step'] == 'Valence Before Shock Odor Right']
odor_left_df_before_shock = df[df['experiment_step'] == 'Valence Before Shock Odor Left']

odor_right_df_after_shock = df[df['experiment_step'] == 'Valence After Shock Odor Right']
odor_left_df_after_shock = df[df['experiment_step'] == 'Valence After Shock Odor Left']

# Filter each experiment step separately
print("Filtering each experiment step...")
valid_chambers_before_right = filter_moving_flies(odor_right_df_before_shock, threshold_X=60, min_crossings_Y=1)
print(f"  Before Shock Odor Right: {len(valid_chambers_before_right)} out of 20 chambers passed")

valid_chambers_before_left = filter_moving_flies(odor_left_df_before_shock, threshold_X=60, min_crossings_Y=1)
print(f"  Before Shock Odor Left: {len(valid_chambers_before_left)} out of 20 chambers passed")

valid_chambers_after_right = filter_moving_flies(odor_right_df_after_shock, threshold_X=60, min_crossings_Y=1)
print(f"  After Shock Odor Right: {len(valid_chambers_after_right)} out of 20 chambers passed")

valid_chambers_after_left = filter_moving_flies(odor_left_df_after_shock, threshold_X=60, min_crossings_Y=1)
print(f"  After Shock Odor Left: {len(valid_chambers_after_left)} out of 20 chambers passed")

# Find union of before pair (OR logic): chambers that passed at least one "before" step
valid_chambers_before_pair = set(valid_chambers_before_right) | set(valid_chambers_before_left)
print(f"  Before pair (right OR left): {len(valid_chambers_before_pair)} chambers passed")

# Find union of after pair (OR logic): chambers that passed at least one "after" step
valid_chambers_after_pair = set(valid_chambers_after_right) | set(valid_chambers_after_left)
print(f"  After pair (right OR left): {len(valid_chambers_after_pair)} chambers passed")

# Find intersection of both pairs (AND logic): chambers that passed at least one "before" AND at least one "after"
valid_chambers_all = valid_chambers_before_pair & valid_chambers_after_pair
valid_chambers_all = sorted(list(valid_chambers_all))  # Convert to sorted list for consistent ordering

print(f"\nChambers that passed (before pair OR) AND (after pair OR): {len(valid_chambers_all)} out of 20")

# For each chamber that passed (before pair OR) AND (after pair OR), perform the calculation
chamber_numbers = []
valence_before = []
valence_after = []
valence_difference = []
for col in valid_chambers_all:
    # Extract chamber number from column name (e.g., "chamber_1_loc" -> 1)
    chamber_num = int(col.split('_')[1])
    chamber_numbers.append(chamber_num)
    # Calculate ratio for odor_right_df: number of rows where value < 0 / total rows
    odor_right_ratio_before_shock = (odor_right_df_before_shock[col] < 0).sum() / len(odor_right_df_before_shock)
    odor_left_ratio_before_shock = (odor_left_df_before_shock[col] < 0).sum() / len(odor_left_df_before_shock)
    odor_right_ratio_after_shock = (odor_right_df_after_shock[col] < 0).sum() / len(odor_right_df_after_shock)
    odor_left_ratio_after_shock = (odor_left_df_after_shock[col] < 0).sum() / len(odor_left_df_after_shock)
    # Calculate valence: subtract odor_left from odor_right and multiply by 100
    before = (odor_right_ratio_before_shock - odor_left_ratio_before_shock) * 100
    after = (odor_right_ratio_after_shock - odor_left_ratio_after_shock) * 100
    difference = after - before
    valence_before.append(before)
    valence_after.append(after)
    valence_difference.append(difference)

# Create result dataframe (only for valid chambers)
result_df = pd.DataFrame({
    'chamber': [f'chamber_{i}' for i in chamber_numbers],
    'valence_before': valence_before,
    'valence_after': valence_after,
    'valence_difference': valence_difference
})

# Export result df to output.csv
result_df.to_csv('output.csv', index=False)

# Calculate and print statistical properties
average = result_df['valence_difference'].mean()
sem = result_df['valence_difference'].std() / np.sqrt(len(result_df))
print(f"\nAverage valence difference: {average}")
print(f"SEM: {sem}")

# Display the DataFrame
print(df.head())
print(f"\nDataFrame shape: {df.shape}")
