import pandas as pd


def filter_moving_flies(df, decision_point, min_decisions_filter):
    """
    Filter out flies that haven't moved enough.
    
    Parameters:
    - df: pandas DataFrame
    - determine_side: integer
    - decision_point: integer
    - min_decisions_filter: integer, minimum total number of crossings required for filtering
    
    Returns:
    - List of chamber column names that passed the filter
    """
    # Extract all column names chamber_X_loc where X is a number between 1 to 20
    chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
    
    # Filter out flies that haven't moved enough
    valid_chambers = []
    
    for col in chamber_columns:
        positions = df[col].dropna().values
        
        if len(positions) < 2:
            continue  # Skip if not enough data
        
        # Count crossings of +X (going from < X to >= X)
        crossings_plus_X = 0
        for i in range(len(positions) - 1):
            if positions[i] < decision_point and positions[i+1] >= decision_point:
                crossings_plus_X += 1
        
        # Count crossings of -X (going from > -X to <= -X)
        crossings_minus_X = 0
        for i in range(len(positions) - 1):
            if positions[i] > -decision_point and positions[i+1] <= -decision_point:
                crossings_minus_X += 1
        
        # Calculate total crossings
        total_crossings = crossings_plus_X + crossings_minus_X
        
        # Check if crossing counts meet threshold
        meets_plus_threshold = crossings_plus_X >= min_decisions_filter
        meets_minus_threshold = crossings_minus_X >= min_decisions_filter
        
        # Keep if either crossing count meets threshold (passed +X OR -X at least Y times)
        # AND total crossings meets minimum decisions filter
        if (meets_plus_threshold or meets_minus_threshold) and total_crossings >= min_decisions_filter:
            valid_chambers.append(col)
    
    return valid_chambers


def calculate_learning(df, decision_side, cs_plus_side, valid_chambers):
    """
    Perform calculation on the dataframe for valid chambers.
    
    Parameters:
    - df: pandas DataFrame
    - decision_point: integer
    - cs_plus_side: string, either "left" or "right"
    - valid_chambers: List of chamber column names to calculate for
    
    Returns:
    - DataFrame with chamber and fraction columns
    """
    # Calculate ratio: number of rows below decision_side / total rows for each valid column
    chamber_numbers = []
    fractions = []
    
    for col in valid_chambers:
        # Extract chamber number from column name (e.g., "chamber_1_loc" -> 1)
        chamber_num = int(col.split('_')[1])
        chamber_numbers.append(chamber_num)
        
        # Calculate fraction: number of rows where value < or > decision_side / total rows
        total_rows = len(df[col].dropna())
        if cs_plus_side == "left":
            rows_below = (df[col] < decision_side).sum()
        elif cs_plus_side == "right":
            rows_below = (df[col] > decision_side).sum()
        fraction = rows_below / total_rows if total_rows > 0 else 0
        fractions.append(fraction)
    
    # Create and return dataframe with fraction values
    result_df = pd.DataFrame({
        'chamber': [f'chamber_{i}' for i in chamber_numbers],
        'fraction': fractions
    })
    
    return result_df

# Load CSV file from classical/fly_loc.csv into a pandas DataFrame
df = pd.read_csv(r"D:\multiplex\system_check\w1118_classical\11.12.2025\trial_7\fly_loc.csv")

# Create two dataframes based on experiment_step
initial_df = df[df['experiment_step'] == 'Initial Valence']
test_df = df[df['experiment_step'] == 'Test']

# Apply filter function to both dataframes separately
initial_valid_chambers = filter_moving_flies(initial_df, decision_point=30, min_decisions_filter=1)
test_valid_chambers = filter_moving_flies(test_df, decision_point=30, min_decisions_filter=1)

# Find intersection of valid chambers (chambers that passed both filters)
valid_chambers_intersection = set(initial_valid_chambers) & set(test_valid_chambers)
valid_chambers_intersection = sorted(list(valid_chambers_intersection))

# Apply calculate_learning function to both dataframes using only the intersection of valid chambers
initial_result = calculate_learning(initial_df, decision_side=30, cs_plus_side="left", valid_chambers=valid_chambers_intersection)
test_result = calculate_learning(test_df, decision_side=30, cs_plus_side="right", valid_chambers=valid_chambers_intersection)

# Merge the results and calculate difference
learning_df = pd.merge(initial_result, test_result, on='chamber', suffixes=('_initial', '_test'))
learning_df['difference'] = learning_df['fraction_test'] - learning_df['fraction_initial']
learning_df = learning_df.rename(columns={'fraction_initial': 'initial_fraction', 'fraction_test': 'test_fraction'})

# Export learning_df to CSV
learning_df.to_csv('classical_learning_output.csv', index=False)
print(f"Exported {len(learning_df)} chambers to classical_learning_output.csv")
print(learning_df)
