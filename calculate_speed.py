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


def per_chamber_speed_visualization(df, speed_df):
    """
    Visualize location and speed for all 20 chambers in a 20-row layout.
    
    Parameters:
    - df: DataFrame with location data
    - speed_df: DataFrame with speed data (from calculate_derivatives)
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create 20 rows of subplots (20 rows, 1 column = 20 chambers)
    # Increased height to 80 for better visibility of each row
    fig, axes = plt.subplots(20, 1, figsize=(16, 80))
    fig.suptitle('All 20 Chambers - Location and Speed', fontsize=16, y=0.995)
    
    sample_indices = range(len(df))
    chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
    
    # Calculate average speed for summary
    all_speeds = []
    
    # Plot each chamber
    for idx, col in enumerate(chamber_columns):
        ax = axes[idx]
        speed_col_name = col.replace('_loc', '_speed')
        
        location_values = df[col].values
        speed_values = speed_df[speed_col_name].values
        all_speeds.extend(speed_values[~np.isnan(speed_values)])
        
        # Plot location on left y-axis
        ax1 = ax
        color1 = '#1f77b4'
        ax1.set_xlabel('Sample Index' if idx == 19 else '', fontsize=10)
        ax1.set_ylabel('Location', color=color1, fontsize=9)
        line1 = ax1.plot(sample_indices, location_values, color=color1, linewidth=1.5, label='Location', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot speed on right y-axis
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        ax2.set_ylabel('Speed', color=color2, fontsize=9)
        line2 = ax2.plot(sample_indices, speed_values, color=color2, linewidth=1.5, label='Speed', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)
        
        # Set title for subplot
        ax1.set_title(f'Chamber {idx + 1}', fontsize=10, fontweight='bold', pad=5)
        
        # Only show x-axis label on the last row
        if idx != 19:
            ax1.set_xlabel('')
    
    # Calculate average speed
    avg_speed = np.mean(all_speeds) if all_speeds else 0
    
    # Add average speed to title
    fig.suptitle(f'All 20 Chambers - Location and Speed (Average Speed: {avg_speed:.2f})', 
                 fontsize=16, y=0.995)
    
    # Add legend
    fig.legend([line1[0], line2[0]], ['Location', 'Speed'], 
              loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save as high-quality image
    output_filename = 'chambers_speed_visualization.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High-quality image saved as: {output_filename}")
    
    plt.close()


def epoch_analysis_visualization(df, speed_df):
    """
    Split dataframe into epochs based on experiment_step values, calculate statistics,
    and visualize mean and SEM for location and speed in barplots.
    
    An epoch is defined as consecutive rows with the same experiment_step value
    (including NaN periods between named steps).
    
    Parameters:
    - df: DataFrame with location data and experiment_step column
    - speed_df: DataFrame with speed data (from calculate_derivatives)
    """
    # Step 1: Identify epochs by finding consecutive groups of the same experiment_step value
    # We'll create a group identifier that changes whenever experiment_step changes
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
    
    # Step 2: Calculate statistics for each epoch
    # For each epoch, calculate mean and SEM across all chambers for both location and speed
    print("\nStep 2: Calculating statistics (mean and SEM) for each epoch...")
    
    chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
    speed_columns = [f'chamber_{i}_speed' for i in range(1, 21)]
    
    epoch_stats = []
    
    for epoch_id in epochs_info['epoch_id']:
        # Get rows for this epoch
        epoch_mask = df_with_epochs['epoch_id'] == epoch_id
        epoch_df = df_with_epochs[epoch_mask]
        epoch_speed_df = speed_df[epoch_mask]
        
        # Get epoch step name
        epoch_step = epochs_info[epochs_info['epoch_id'] == epoch_id]['epoch_step'].values[0]
        
        # Calculate mean per chamber for this epoch, then calculate SEM across chambers
        # This treats each chamber as a sample, which is more appropriate
        chamber_location_means = []
        chamber_speed_means = []
        
        for col in chamber_columns:
            # Get location values for this chamber in this epoch (excluding NaN)
            loc_values = epoch_df[col].dropna().values
            if len(loc_values) > 0:
                chamber_location_means.append(np.mean(loc_values))
            
            # Get speed values for this chamber in this epoch (excluding NaN)
            speed_col = col.replace('_loc', '_speed')
            speed_values = epoch_speed_df[speed_col].dropna().values
            if len(speed_values) > 0:
                chamber_speed_means.append(np.mean(speed_values))
        
        # Calculate overall mean and SEM across chambers
        if len(chamber_location_means) > 0:
            location_mean = np.mean(chamber_location_means)
            location_sem = np.std(chamber_location_means, ddof=1) / np.sqrt(len(chamber_location_means))
        else:
            location_mean = np.nan
            location_sem = np.nan
        
        if len(chamber_speed_means) > 0:
            speed_mean = np.mean(chamber_speed_means)
            speed_sem = np.std(chamber_speed_means, ddof=1) / np.sqrt(len(chamber_speed_means))
        else:
            speed_mean = np.nan
            speed_sem = np.nan
        
        epoch_stats.append({
            'epoch_id': epoch_id,
            'epoch_step': epoch_step,
            'location_mean': location_mean,
            'location_sem': location_sem,
            'speed_mean': speed_mean,
            'speed_sem': speed_sem,
            'n_chambers': len(chamber_location_means)
        })
        
        print(f"  Epoch {epoch_id} ('{epoch_step}'): Location={location_mean:.2f}±{location_sem:.2f}, "
              f"Speed={speed_mean:.2f}±{speed_sem:.2f} (n={len(chamber_location_means)} chambers)")
    
    # Step 3: Create visualization with two barplots
    print("\nStep 3: Creating barplots for location and speed...")
    
    stats_df = pd.DataFrame(epoch_stats)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for plotting
    x_pos = np.arange(len(stats_df))
    epoch_labels = [f"Epoch {i+1}\n({row['epoch_step']})" for i, row in stats_df.iterrows()]
    
    # Plot 1: Location barplot
    location_means = stats_df['location_mean'].values
    location_sems = stats_df['location_sem'].values
    
    bars1 = ax1.bar(x_pos, location_means, yerr=location_sems, capsize=5, 
                    color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Location (Mean ± SEM)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Location by Epoch', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(epoch_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(location_means, location_sems)):
        if not np.isnan(mean):
            ax1.text(i, mean + sem + abs(mean) * 0.02, f'{mean:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Speed barplot
    speed_means = stats_df['speed_mean'].values
    speed_sems = stats_df['speed_sem'].values
    
    bars2 = ax2.bar(x_pos, speed_means, yerr=speed_sems, capsize=5, 
                    color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speed (Mean ± SEM)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Speed by Epoch', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(epoch_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(speed_means, speed_sems)):
        if not np.isnan(mean):
            ax2.text(i, mean + sem + abs(mean) * 0.02, f'{mean:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save as high-quality image
    output_filename = 'epoch_analysis_visualization.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nHigh-quality image saved as: {output_filename}")
    
    plt.close()
    
    # Return the statistics dataframe for further analysis if needed
    return stats_df


# Read CSV file as input
df = pd.read_csv(r"D:\multiplex\system_check\w1118_classical\11.12.2025\trial_2\fly_loc.csv")

# Create a dataframe with only experiment_step and chamber_x_loc columns (x = 1 to 20)
columns_to_select = ['experiment_step'] + [f'chamber_{i}_loc' for i in range(1, 21)]
df = df[columns_to_select]

# Calculate derivatives for all chamber columns
speed_df = calculate_derivatives(df)

# Visualize all chambers in a grid
per_chamber_speed_visualization(df, speed_df)

# Analyze and visualize epochs
epoch_stats = epoch_analysis_visualization(df, speed_df)
