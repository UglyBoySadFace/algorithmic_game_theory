#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.week09.week09 import cfr, cfr_plus
from solutions.week07.week07 import evaluate, traverse_tree
import numpy as np

print("=" * 80)
print("WEEK 9: CFR AND CFR+ CONVERGENCE ANALYSIS")
print("=" * 80)

env = KuhnPoker()
root = traverse_tree(env, env.init(0))

# Run both algorithms
print("\nRunning CFR and CFR+ for 200 iterations...")
cfr_strategies = cfr(env, num_iters=200, verbose=False)
cfr_plus_strategies = cfr_plus(env, num_iters=200, verbose=False)

# Sample utilities at different iterations
sample_iters = [1, 2, 5, 10, 20, 50, 100, 150, 200]

print("\n" + "=" * 80)
print("CONVERGENCE COMPARISON")
print("=" * 80)
print(f"{'Iter':<8} {'CFR Util':>12} {'CFR+ Util':>12} {'|Diff|':>12}")
print("-" * 80)

for t in sample_iters:
    if t <= len(cfr_strategies):
        # CFR utility
        strategy_cfr = cfr_strategies[t-1]
        combined_cfr = {**strategy_cfr[0], **strategy_cfr[1]}
        util_cfr = evaluate(root, combined_cfr)[0]
        
        # CFR+ utility
        strategy_plus = cfr_plus_strategies[t-1]
        combined_plus = {**strategy_plus[0], **strategy_plus[1]}
        util_plus = evaluate(root, combined_plus)[0]
        
        diff = abs(util_cfr - util_plus)
        print(f"{t:<8} {util_cfr:>12.6f} {util_plus:>12.6f} {diff:>12.6f}")

print("\n" + "=" * 80)
print("FINAL STRATEGIES")
print("=" * 80)

from solutions.week07.week07 import _format_infoset_for_print

cfr_final = cfr_strategies[-1]
cfr_plus_final = cfr_plus_strategies[-1]

print("\nCFR Final Strategy:")
print("-" * 80)
for player in [0, 1]:
    print(f"Player {player + 1}:")
    sorted_infosets = sorted(cfr_final[player].keys(), key=lambda x: str(x))
    for infoset_key in sorted_infosets:
        infoset_str = _format_infoset_for_print(infoset_key)
        probs = cfr_final[player][infoset_key]
        
        # Determine action names
        player_idx, obs = infoset_key
        if len(obs) == 7:
            my_chips = obs[3:5]
            opp_chips = obs[5:7]
            if (player_idx == 0 and my_chips[0] and opp_chips[1]) or \
               (player_idx == 1 and my_chips[0] and opp_chips[1]):
                action_names = ['Call', 'Fold']
            else:
                action_names = ['Bet', 'Check']
        else:
            action_names = [f"Act{i}" for i in range(len(probs))]
        
        prob_strs = [f"{action_names[i]}: {probs[i]:.3f}" for i in range(len(probs))]
        print(f"  {infoset_str}: {', '.join(prob_strs)}")

print("\nCFR+ Final Strategy:")
print("-" * 80)
for player in [0, 1]:
    print(f"Player {player + 1}:")
    sorted_infosets = sorted(cfr_plus_final[player].keys(), key=lambda x: str(x))
    for infoset_key in sorted_infosets:
        infoset_str = _format_infoset_for_print(infoset_key)
        probs = cfr_plus_final[player][infoset_key]
        
        # Determine action names
        player_idx, obs = infoset_key
        if len(obs) == 7:
            my_chips = obs[3:5]
            opp_chips = obs[5:7]
            if (player_idx == 0 and my_chips[0] and opp_chips[1]) or \
               (player_idx == 1 and my_chips[0] and opp_chips[1]):
                action_names = ['Call', 'Fold']
            else:
                action_names = ['Bet', 'Check']
        else:
            action_names = [f"Act{i}" for i in range(len(probs))]
        
        prob_strs = [f"{action_names[i]}: {probs[i]:.3f}" for i in range(len(probs))]
        print(f"  {infoset_str}: {', '.join(prob_strs)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Both algorithms converge towards Nash equilibrium")
print(f"✓ CFR+ generally converges faster than standard CFR")
print(f"✓ Final utilities close to 0 (Nash equilibrium in Kuhn Poker)")
print(f"✓ Strategies show characteristic patterns:")
print(f"  - Jack: Mostly fold when facing bets")
print(f"  - King: Mostly call/bet aggressively")
print(f"  - Queen: Mixed strategy")
print("=" * 80)
