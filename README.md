# Valence Analysis

This project analyzes fly location data from CSV files to calculate valence preferences between odor right and odor left conditions.

## Description

The script processes fly location data by:
1. Loading a CSV file containing experiment data
2. Filtering for `experiment_step` and `chamber_x_loc` columns (where x = 1 to 20)
3. Splitting data into two dataframes based on experiment step: `Odor Right` and `Odor Left`
4. Calculating the ratio of rows with values < 0 for each chamber in both conditions
5. Computing the difference between conditions (right - left) multiplied by 100
6. Exporting results to `output.csv`
7. Calculating and printing statistical properties (average and SEM)

## Requirements

- Python 3.x
- pandas
- numpy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Update the CSV file path in `load_csv.py` and run:

```bash
python load_csv.py
```

The script will:
- Process the data
- Generate `output.csv` with results for each chamber
- Print the average and SEM of the results

## Output

The script generates `output.csv` with two columns:
- `chamber`: Chamber identifier (chamber_1 through chamber_20)
- `result`: Calculated difference value for each chamber

Statistical summary (average and SEM) is printed to the console.
