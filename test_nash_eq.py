#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.week08.week08 import (
    find_nash_equilibrium_sequence_form,
    convert_realization_plan_to_behavioural_strategy
)
from solutions.week07.week07 import _format_infoset_for_print
import numpy as np

print("=" * 80)
print("NASH EQUILIBRIUM VIA SEQUENCE-FORM LP")
print("=" * 80)

env = KuhnPoker()

# Find Nash equilibrium
r_p0, r_p1 = find_nash_equilibrium_sequence_form(env, seed=0)

# Convert to behavioral strategies
behavioral_p0 = convert_realization_plan_to_behavioural_strategy(env, 0, r_p0, seed=0)
behavioral_p1 = convert_realization_plan_to_behavioural_strategy(env, 1, r_p1, seed=0)

print("\nPlayer 0 Nash Equilibrium Strategy:")
print("-" * 80)
for infoset_key in sorted(behavioral_p0.keys(), key=lambda x: str(x)):
    infoset_str = _format_infoset_for_print(infoset_key)
    probs = behavioral_p0[infoset_key]
    
    # Determine action names based on the information set
    player, obs = infoset_key
    if len(obs) == 7:
        my_chips = obs[3:5]
        opp_chips = obs[5:7]
        
        if player == 0 and my_chips[0] and opp_chips[1]:
            action_names = ['Call', 'Fold']
        elif player == 1 and my_chips[0] and opp_chips[1]:
            action_names = ['Call', 'Fold']
        else:
            action_names = ['Bet', 'Check']
    else:
        action_names = [f"Action{i}" for i in range(len(probs))]
    
    prob_strs = [f"{action_names[i]}: {probs[i]:.5f}" for i in range(len(probs))]
    print(f"  {infoset_str}: {', '.join(prob_strs)}")

print("\nPlayer 1 Nash Equilibrium Strategy:")
print("-" * 80)
for infoset_key in sorted(behavioral_p1.keys(), key=lambda x: str(x)):
    infoset_str = _format_infoset_for_print(infoset_key)
    probs = behavioral_p1[infoset_key]
    
    # Determine action names
    player, obs = infoset_key
    if len(obs) == 7:
        my_chips = obs[3:5]
        opp_chips = obs[5:7]
        
        if player == 0 and my_chips[0] and opp_chips[1]:
            action_names = ['Call', 'Fold']
        elif player == 1 and my_chips[0] and opp_chips[1]:
            action_names = ['Call', 'Fold']
        else:
            action_names = ['Bet', 'Check']
    else:
        action_names = [f"Action{i}" for i in range(len(probs))]
    
    prob_strs = [f"{action_names[i]}: {probs[i]:.5f}" for i in range(len(probs))]
    print(f"  {infoset_str}: {', '.join(prob_strs)}")

# Compute game value
from solutions.week08.week08 import convert_to_sequence_form
F_p0, _, _, _, _, _ = convert_to_sequence_form(env, seed=0)
game_value = r_p0.T @ F_p0 @ r_p1

print("\n" + "=" * 80)
print(f"Game Value: {game_value:.5f}")
print("=" * 80)

# Compare with known optimal Kuhn Poker strategy
print("\nKnown Optimal Kuhn Poker Strategy (approximate):")
print("-" * 80)
print("P1 with Jack: Always check/fold")
print("P1 with Queen: Check, call if opponent bets") 
print("P1 with King: Bet ~3x pot, call if checked to")
print("P2: Mirror strategy with information about P1's action")
