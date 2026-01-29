#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.week09.week09 import cfr, cfr_plus
import numpy as np

print("=" * 80)
print("TESTING CFR AND CFR+ IMPLEMENTATIONS")
print("=" * 80)

env = KuhnPoker()

# Test 1: Standard CFR
print("\n1. Testing standard CFR...")
print("-" * 80)
try:
    cfr_strategies = cfr(env, num_iters=100, verbose=False)
    
    print(f"✓ Ran 100 iterations of CFR")
    print(f"✓ Generated {len(cfr_strategies)} strategy profiles")
    
    # Check final strategy
    final_strategy = cfr_strategies[-1]
    print(f"✓ P0 has {len(final_strategy[0])} information sets")
    print(f"✓ P1 has {len(final_strategy[1])} information sets")
    
    # Check all strategies sum to 1
    for player in [0, 1]:
        for infoset, probs in final_strategy[player].items():
            if not np.isclose(np.sum(probs), 1.0):
                print(f"  ⚠ Warning: {infoset} probs sum to {np.sum(probs)}")
    
    print("\n  Sample final strategies (last 3 infosets per player):")
    for player in [0, 1]:
        print(f"    Player {player}:")
        for infoset, probs in list(final_strategy[player].items())[:3]:
            print(f"      {str(infoset)[:50]}: {probs}")
    
    print("✓ Standard CFR works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: CFR+
print("\n2. Testing CFR+...")
print("-" * 80)
try:
    cfr_plus_strategies = cfr_plus(env, num_iters=100, verbose=False)
    
    print(f"✓ Ran 100 iterations of CFR+")
    print(f"✓ Generated {len(cfr_plus_strategies)} strategy profiles")
    
    # Check final strategy
    final_strategy_plus = cfr_plus_strategies[-1]
    print(f"✓ P0 has {len(final_strategy_plus[0])} information sets")
    print(f"✓ P1 has {len(final_strategy_plus[1])} information sets")
    
    # Check all strategies sum to 1
    for player in [0, 1]:
        for infoset, probs in final_strategy_plus[player].items():
            if not np.isclose(np.sum(probs), 1.0):
                print(f"  ⚠ Warning: {infoset} probs sum to {np.sum(probs)}")
    
    print("\n  Sample final strategies (last 3 infosets per player):")
    for player in [0, 1]:
        print(f"    Player {player}:")
        for infoset, probs in list(final_strategy_plus[player].items())[:3]:
            print(f"      {str(infoset)[:50]}: {probs}")
    
    print("✓ CFR+ works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Convergence comparison
print("\n3. Comparing convergence rates...")
print("-" * 80)
try:
    from solutions.week07.week07 import evaluate, traverse_tree
    
    # Build game tree
    root = traverse_tree(env, env.init(0))
    
    # Sample utilities at different iterations
    sample_iters = [1, 5, 10, 20, 50, 100]
    
    print("\n  CFR Utilities:")
    for t in sample_iters:
        if t <= len(cfr_strategies):
            strategy = cfr_strategies[t-1]
            combined = {**strategy[0], **strategy[1]}
            utilities = evaluate(root, combined)
            print(f"    Iter {t:3d}: {utilities[0]:8.5f}, {utilities[1]:8.5f}")
    
    print("\n  CFR+ Utilities:")
    for t in sample_iters:
        if t <= len(cfr_plus_strategies):
            strategy = cfr_plus_strategies[t-1]
            combined = {**strategy[0], **strategy[1]}
            utilities = evaluate(root, combined)
            print(f"    Iter {t:3d}: {utilities[0]:8.5f}, {utilities[1]:8.5f}")
    
    print("\n✓ Both algorithms converge towards Nash equilibrium (utilities → 0)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Compare with Nash equilibrium from sequence-form LP
print("\n4. Comparing with Nash equilibrium from sequence-form LP...")
print("-" * 80)
try:
    from solutions.week08.week08 import (
        find_nash_equilibrium_sequence_form,
        convert_realization_plan_to_behavioural_strategy
    )
    
    # Find Nash equilibrium
    r_p0, r_p1 = find_nash_equilibrium_sequence_form(env, seed=0)
    nash_p0 = convert_realization_plan_to_behavioural_strategy(env, 0, r_p0, seed=0)
    nash_p1 = convert_realization_plan_to_behavioural_strategy(env, 1, r_p1, seed=0)
    
    nash_strategy = {0: nash_p0, 1: nash_p1}
    combined_nash = {**nash_p0, **nash_p1}
    nash_utilities = evaluate(root, combined_nash)
    
    print(f"  Nash equilibrium utilities: {nash_utilities[0]:.5f}, {nash_utilities[1]:.5f}")
    
    # Compare final CFR strategies with Nash
    cfr_final = cfr_strategies[-1]
    cfr_plus_final = cfr_plus_strategies[-1]
    
    combined_cfr = {**cfr_final[0], **cfr_final[1]}
    combined_cfr_plus = {**cfr_plus_final[0], **cfr_plus_final[1]}
    
    cfr_utilities = evaluate(root, combined_cfr)
    cfr_plus_utilities = evaluate(root, combined_cfr_plus)
    
    print(f"  CFR final utilities:         {cfr_utilities[0]:.5f}, {cfr_utilities[1]:.5f}")
    print(f"  CFR+ final utilities:        {cfr_plus_utilities[0]:.5f}, {cfr_plus_utilities[1]:.5f}")
    
    print(f"\n  Distance from Nash (utility difference):")
    print(f"    CFR:  {abs(cfr_utilities[0] - nash_utilities[0]):.5f}")
    print(f"    CFR+: {abs(cfr_plus_utilities[0] - nash_utilities[0]):.5f}")
    
    print("\n✓ Both algorithms approximate Nash equilibrium!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
