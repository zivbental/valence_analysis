import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def calculate_derivatives(df):
    """
    Calculate derivatives (speed) for all chamber_x_loc columns.
    
    Parameters:
    - df: pandas DataFrame with chamber_x_loc columns
    
    Returns:
    - DataFrame with speed columns (chamber_x_speed)
    """
    speed_df = df.copy()
    chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
    
    for col in chamber_columns:
        # Calculate derivative (difference between consecutive rows) and take absolute value
        speed_col_name = col.replace('_loc', '_speed')
        speed_df[speed_col_name] = np.abs(df[col].diff())
    
    return speed_df


def per_fly_epoch_analysis_visualization(df, speed_df):
    """
    Split dataframe into epochs based on experiment_step values, calculate average speed
    for each individual fly (chamber) per epoch, and visualize the results.
    
    An epoch is defined as consecutive rows with the same experiment_step value
    (including NaN periods between named steps).
    
    Parameters:
    - df: DataFrame with location data and experiment_step column
    - speed_df: DataFrame with speed data (from calculate_derivatives)
    """
    # Step 1: Identify epochs by finding consecutive groups of the same experiment_step value
    print("Step 1: Identifying epochs based on experiment_step values...")
    
    # Fill NaN with a placeholder string to identify gaps
    df_with_epochs = df.copy()
    df_with_epochs['experiment_step_filled'] = df_with_epochs['experiment_step'].fillna('_GAP_')
    
    # Create epoch groups: consecutive rows with the same experiment_step value
    # A new epoch starts when the value changes from the previous row
    df_with_epochs['epoch_id'] = (df_with_epochs['experiment_step_filled'] != 
                                   df_with_epochs['experiment_step_filled'].shift()).cumsum()
    
    # Get unique epochs with their step names
    epochs_info = df_with_epochs.groupby('epoch_id')['experiment_step'].first().reset_index()
    epochs_info['epoch_step'] = epochs_info['experiment_step'].fillna('Gap')
    epochs_info['epoch_size'] = df_with_epochs.groupby('epoch_id').size().values
    
    print(f"Found {len(epochs_info)} epochs")
    for idx, row in epochs_info.iterrows():
        print(f"  Epoch {idx + 1}: '{row['epoch_step']}' ({row['epoch_size']} rows)")
    
    # Step 2: Calculate average speed for each individual fly (chamber) for each epoch
    print("\nStep 2: Calculating average speed for each individual fly per epoch...")
    
    chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
    
    # Store results: each row will be (chamber_number, epoch_id, epoch_step, average_speed)
    per_fly_epoch_stats = []
    
    for epoch_id in epochs_info['epoch_id']:
        # Get rows for this epoch
        epoch_mask = df_with_epochs['epoch_id'] == epoch_id
        epoch_speed_df = speed_df[epoch_mask]
        
        # Get epoch step name
        epoch_step = epochs_info[epochs_info['epoch_id'] == epoch_id]['epoch_step'].values[0]
        
        # Calculate average speed for each individual chamber (fly) in this epoch
        for chamber_num in range(1, 21):
            speed_col = f'chamber_{chamber_num}_speed'
            
            # Get speed values for this chamber in this epoch (excluding NaN)
            speed_values = epoch_speed_df[speed_col].dropna().values
            
            if len(speed_values) > 0:
                avg_speed = np.mean(speed_values)
                per_fly_epoch_stats.append({
                    'chamber': chamber_num,
                    'epoch_id': epoch_id,
                    'epoch_step': epoch_step,
                    'average_speed': avg_speed
                })
                print(f"  Chamber {chamber_num}, Epoch {epoch_id} ('{epoch_step}'): Average Speed = {avg_speed:.2f}")
            else:
                per_fly_epoch_stats.append({
                    'chamber': chamber_num,
                    'epoch_id': epoch_id,
                    'epoch_step': epoch_step,
                    'average_speed': np.nan
                })
    
    # Step 3: Create visualization
    print("\nStep 3: Creating visualization for per-fly average speed by epoch...")
    
    stats_df = pd.DataFrame(per_fly_epoch_stats)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create a pivot table for easier visualization: chambers as rows, epochs as columns
    pivot_df = stats_df.pivot_table(
        index='chamber',
        columns='epoch_id',
        values='average_speed',
        aggfunc='first'
    )
    
    # Get epoch labels for column names
    epoch_labels = []
    for epoch_id in pivot_df.columns:
        epoch_step = epochs_info[epochs_info['epoch_id'] == epoch_id]['epoch_step'].values[0]
        epoch_labels.append(f"Epoch {epoch_id}\n({epoch_step})")
    
    # Create figure with heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(pivot_df.columns) * 1.5), 10))
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Average Speed'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Chamber (Fly)', fontsize=12, fontweight='bold')
    ax.set_title('Average Speed per Individual Fly by Epoch', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticklabels(epoch_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([f'Chamber {i}' for i in pivot_df.index], fontsize=9)
    
    plt.tight_layout()
    
    # Save as high-quality image
    output_filename = 'per_fly_epoch_speed_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nHigh-quality image saved as: {output_filename}")
    
    plt.close()
    
    # Also create a line plot showing each fly's speed across epochs
    fig, ax = plt.subplots(figsize=(max(12, len(epochs_info) * 1.5), 8))
    
    # Plot each fly's speed across epochs
    for chamber_num in range(1, 21):
        chamber_data = stats_df[stats_df['chamber'] == chamber_num].sort_values('epoch_id')
        if not chamber_data['average_speed'].isna().all():
            ax.plot(chamber_data['epoch_id'], chamber_data['average_speed'], 
                   marker='o', label=f'Chamber {chamber_num}', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Epoch ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Speed', fontsize=12, fontweight='bold')
    ax.set_title('Average Speed per Individual Fly Across Epochs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    # Save line plot
    output_filename2 = 'per_fly_epoch_speed_lineplot.png'
    plt.savefig(output_filename2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High-quality line plot saved as: {output_filename2}")
    
    plt.close()
    
    # Create a barplot showing average speed across all flies for each epoch (with SEM)
    print("\nStep 4: Creating barplot with average speed across all flies per epoch...")
    
    # Calculate mean and SEM across all flies for each epoch
    epoch_summary = []
    for epoch_id in epochs_info['epoch_id']:
        epoch_step = epochs_info[epochs_info['epoch_id'] == epoch_id]['epoch_step'].values[0]
        epoch_speeds = stats_df[stats_df['epoch_id'] == epoch_id]['average_speed'].dropna().values
        
        if len(epoch_speeds) > 0:
            mean_speed = np.mean(epoch_speeds)
            if len(epoch_speeds) == 1:
                sem_speed = 0.0  # No variance with single sample
            else:
                sem_speed = np.std(epoch_speeds, ddof=1) / np.sqrt(len(epoch_speeds))
            epoch_summary.append({
                'epoch_id': epoch_id,
                'epoch_step': epoch_step,
                'mean_speed': mean_speed,
                'sem_speed': sem_speed,
                'n_flies': len(epoch_speeds)
            })
        else:
            epoch_summary.append({
                'epoch_id': epoch_id,
                'epoch_step': epoch_step,
                'mean_speed': np.nan,
                'sem_speed': np.nan,
                'n_flies': 0
            })
    
    summary_df = pd.DataFrame(epoch_summary)
    
    # Create barplot
    fig, ax = plt.subplots(figsize=(max(12, len(summary_df) * 1.2), 6))
    
    x_pos = np.arange(len(summary_df))
    epoch_labels = [f"Epoch {i+1}\n({row['epoch_step']})" for i, row in summary_df.iterrows()]
    
    means = summary_df['mean_speed'].values
    sems = summary_df['sem_speed'].values
    
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                  color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Speed (Mean ± SEM)', fontsize=12, fontweight='bold')
    ax.set_title('Average Speed Across All Flies by Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(epoch_labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(means, sems)):
        if not np.isnan(mean):
            ax.text(i, mean + sem + abs(mean) * 0.02, f'{mean:.2f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save barplot
    output_filename3 = 'average_speed_all_flies_by_epoch.png'
    plt.savefig(output_filename3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High-quality barplot saved as: {output_filename3}")
    
    plt.close()
    
    # Save data to CSV (long format: each row is a chamber-epoch combination)
    csv_filename = 'per_fly_epoch_speed_data.csv'
    stats_df.to_csv(csv_filename, index=False)
    print(f"Data saved to CSV (long format): {csv_filename}")
    
    # Also save in pivot format (chambers as rows, epochs as columns) for easier reading
    pivot_csv_filename = 'per_fly_epoch_speed_pivot.csv'
    pivot_df.to_csv(pivot_csv_filename)
    print(f"Data saved to CSV (pivot format): {pivot_csv_filename}")
    
    # Save summary statistics (mean and SEM across all flies per epoch)
    summary_csv_filename = 'average_speed_all_flies_summary.csv'
    summary_df.to_csv(summary_csv_filename, index=False)
    print(f"Summary statistics saved to CSV: {summary_csv_filename}")
    
    # Return the statistics dataframe for further analysis if needed
    return stats_df


# Read CSV file as input
df = pd.read_csv(r"D:\multiplex\system_check\chamber_shock_Test\18.12.2025\trial_1\fly_loc.csv")

# Create a dataframe with only experiment_step and chamber_x_loc columns (x = 1 to 20)
columns_to_select = ['experiment_step'] + [f'chamber_{i}_loc' for i in range(1, 21)]
df = df[columns_to_select]

# Calculate derivatives for all chamber columns
speed_df = calculate_derivatives(df)

# Analyze and visualize epochs with per-fly averages
per_fly_stats = per_fly_epoch_analysis_visualization(df, speed_df)
