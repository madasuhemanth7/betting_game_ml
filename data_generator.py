# data_generator.py
import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_matches=1000, filename="match_data.csv"):
    """
    Generates a synthetic dataset for sports match outcomes.

    Args:
        num_matches (int): The number of match records to generate.
        filename (str): The name of the CSV file to save the data.
    """
    print(f"Generating {num_matches} synthetic match records...")

    # Generate features
    team_a_form = np.random.normal(loc=0.5, scale=0.2, size=num_matches) # Recent performance, 0-1 scale
    team_b_form = np.random.normal(loc=0.5, scale=0.2, size=num_matches)
    team_a_win_rate = np.random.normal(loc=0.6, scale=0.15, size=num_matches) # Historical win rate, 0-1 scale
    team_b_win_rate = np.random.normal(loc=0.6, scale=0.15, size=num_matches)

    # Clip values to be within a reasonable range (e.g., 0 to 1)
    team_a_form = np.clip(team_a_form, 0, 1)
    team_b_form = np.clip(team_b_form, 0, 1)
    team_a_win_rate = np.clip(team_a_win_rate, 0, 1)
    team_b_win_rate = np.clip(team_b_win_rate, 0, 1)

    # Calculate a "strength difference"
    # Team A is stronger if this value is positive, Team B if negative
    strength_difference = (team_a_form - team_b_form) * 0.6 + \
                          (team_a_win_rate - team_b_win_rate) * 0.4

    # Introduce some randomness to the outcome
    # Higher strength difference means higher probability of Team A winning
    probabilities_team_a_win = 1 / (1 + np.exp(-10 * strength_difference)) # Sigmoid function to map to 0-1

    # Generate outcomes based on probabilities
    outcomes = (np.random.rand(num_matches) < probabilities_team_a_win).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'team_a_form': team_a_form,
        'team_b_form': team_b_form,
        'team_a_win_rate': team_a_win_rate,
        'team_b_win_rate': team_b_win_rate,
        'outcome': outcomes # 1 if Team A wins, 0 if Team B wins
    })

    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(data.head()) # Display first few rows

if __name__ == "__main__":
    generate_synthetic_data()