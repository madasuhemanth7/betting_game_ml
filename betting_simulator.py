# betting_simulator.py
import pandas as pd
import numpy as np
import joblib # For loading models
import os

def simulate_match_outcome(team_a_form, team_b_form, team_a_win_rate, team_b_win_rate):
    """
    Simulates the actual outcome of a match based on team stats.
    This function should ideally be independent of the data generation process,
    but for simplicity, it uses a similar probabilistic approach.

    Args:
        team_a_form (float): Form of Team A.
        team_b_form (float): Form of Team B.
        team_a_win_rate (float): Win rate of Team A.
        team_b_win_rate (float): Win rate of Team B.

    Returns:
        int: 1 if Team A wins, 0 if Team B wins.
    """
    strength_difference = (team_a_form - team_b_form) * 0.6 + \
                          (team_a_win_rate - team_b_win_rate) * 0.4
    probability_team_a_win = 1 / (1 + np.exp(-10 * strength_difference))
    return int(np.random.rand() < probability_team_a_win)

def run_betting_simulation(num_simulations=100, model_filename="betting_model.pkl"):
    """
    Runs a simulation of betting using the trained ML model.

    Args:
        num_simulations (int): The number of matches to simulate bets on.
        model_filename (str): The path to the saved ML model.
    """
    if not os.path.exists(model_filename):
        print(f"Error: Model file '{model_filename}' not found. Please run ml_predictor.py first.")
        return

    print(f"Loading ML model from {model_filename}...")
    model = joblib.load(model_filename)
    print("Model loaded successfully.")

    total_profit_loss = 0
    bet_amount = 10 # Amount to bet on each match

    print(f"\n--- Starting Betting Simulation ({num_simulations} matches) ---")

    for i in range(num_simulations):
        # Generate new, unseen match statistics for simulation
        current_team_a_form = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
        current_team_b_form = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
        current_team_a_win_rate = np.clip(np.random.normal(loc=0.6, scale=0.15), 0, 1)
        current_team_b_win_rate = np.clip(np.random.normal(loc=0.6, scale=0.15), 0, 1)

        # Prepare features for prediction
        match_features = pd.DataFrame([[
            current_team_a_form,
            current_team_b_form,
            current_team_a_win_rate,
            current_team_b_win_rate
        ]], columns=['team_a_form', 'team_b_form', 'team_a_win_rate', 'team_b_win_rate'])

        # Predict outcome using the ML model
        predicted_outcome = model.predict(match_features)[0] # 1 for Team A, 0 for Team B

        # Simulate the actual outcome of the match
        actual_outcome = simulate_match_outcome(
            current_team_a_form,
            current_team_b_form,
            current_team_a_win_rate,
            current_team_b_win_rate
        )

        profit_loss = 0
        bet_on_team = "Team A" if predicted_outcome == 1 else "Team B"
        actual_winner = "Team A" if actual_outcome == 1 else "Team B"

        if predicted_outcome == actual_outcome:
            profit_loss = bet_amount # Win the bet
            print(f"Match {i+1}: Bet on {bet_on_team} (Predicted: {predicted_outcome}). Actual winner: {actual_winner} (Outcome: {actual_outcome}). Result: WIN! (+${bet_amount})")
        else:
            profit_loss = -bet_amount # Lose the bet
            print(f"Match {i+1}: Bet on {bet_on_team} (Predicted: {predicted_outcome}). Actual winner: {actual_winner} (Outcome: {actual_outcome}). Result: LOSS! (-${bet_amount})")

        total_profit_loss += profit_loss

    print("\n--- Simulation Complete ---")
    print(f"Total Profit/Loss after {num_simulations} simulations: ${total_profit_loss}")

if __name__ == "__main__":
    run_betting_simulation(num_simulations=500) # You can change the number of simulations here