import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_monte_carlo(num_trials):
    wins_with_switch = 0
    wins_without_switch = 0
    ratio_with_switch = []
    ratio_without_switch = []
    
    for i in range(1, num_trials + 1):
        # Prize door and chosen door are randomly selected
        prize_door = np.random.randint(0, 3)
        chosen_door = np.random.randint(0, 3)

        # Determine the door to reveal and the door to switch to
        reveal_doors = [door for door in range(3) if door != chosen_door and door != prize_door]
        door_revealed = np.random.choice(reveal_doors)
        switch_door = [door for door in range(3) if door != chosen_door and door != door_revealed][0]

        # Count wins with and without switching
        if chosen_door == prize_door:
            wins_without_switch += 1
        if switch_door == prize_door:
            wins_with_switch += 1

        # Store ratios every 100 trials
        if i % 100 == 0:
            ratio_with_switch.append(wins_with_switch / i)
            ratio_without_switch.append(wins_without_switch / i)

    return ratio_with_switch, ratio_without_switch

def run():
    st.set_page_config(page_title="Monte Carlo Simulation - Three Doors Problem", page_icon="🚪")
    st.title("Monte Carlo Simulation: Three Doors Problem")
    
    num_trials = st.number_input("Number of trials", min_value=1000, value=10000, step=1000)
    
    if st.button("Run Simulation"):
        ratio_with_switch, ratio_without_switch = simulate_monte_carlo(num_trials)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(100, num_trials + 1, 100), ratio_with_switch, label='Win Ratio with Switching')
        plt.plot(range(100, num_trials + 1, 100), ratio_without_switch, label='Win Ratio without Switching')
        plt.xlabel('Number of Trials')
        plt.ylabel('Win Ratio')
        plt.title('Win Ratio Over Trials')
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    run()
