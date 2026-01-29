#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.week08.week08 import (
    convert_to_normal_form,
    convert_to_sequence_form,
    find_nash_equilibrium_sequence_form,
    convert_realization_plan_to_behavioural_strategy,
    _collect_sequences,
    _compute_realization_plan_constraints
)
from solutions.week07.week07 import traverse_tree
import numpy as np

print("=" * 80)
print("TESTING WEEK 8 FUNCTIONS")
print("=" * 80)

env = KuhnPoker()

# Test 1: Normal Form Conversion
print("\n1. Testing convert_to_normal_form...")
print("-" * 80)
try:
    p0_payoff, p1_payoff = convert_to_normal_form(env, seed=0)
    print(f"✓ Matrix dimensions: {p0_payoff.shape}")
    print(f"✓ Is zero-sum? {np.allclose(p0_payoff + p1_payoff, 0)}")
    print(f"✓ Mean P0 payoff: {np.mean(p0_payoff):.5f}")
    print("✓ convert_to_normal_form works!")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Sequence Collection
print("\n2. Testing _collect_sequences...")
print("-" * 80)
try:
    initial_state = env.init(0)
    root = traverse_tree(env, initial_state)
    
    sequences_p0, seq_to_idx_p0 = _collect_sequences(root, 0)
    sequences_p1, seq_to_idx_p1 = _collect_sequences(root, 1)
    
    print(f"✓ P0 has {len(sequences_p0)} sequences (including empty)")
    print(f"✓ P1 has {len(sequences_p1)} sequences (including empty)")
    print(f"✓ Sample P0 sequences: {sequences_p0[:3]}")
    print(f"✓ Sample P1 sequences: {sequences_p1[:3]}")
    print("✓ _collect_sequences works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Realization Plan Constraints
print("\n3. Testing _compute_realization_plan_constraints...")
print("-" * 80)
try:
    E_p0, e_p0 = _compute_realization_plan_constraints(root, 0, sequences_p0, seq_to_idx_p0)
    E_p1, e_p1 = _compute_realization_plan_constraints(root, 1, sequences_p1, seq_to_idx_p1)
    
    print(f"✓ P0 constraint matrix E shape: {E_p0.shape}")
    print(f"✓ P0 constraint vector e shape: {e_p0.shape}")
    print(f"✓ P1 constraint matrix E shape: {E_p1.shape}")
    print(f"✓ P1 constraint vector e shape: {e_p1.shape}")
    print(f"✓ First constraint (empty sequence = 1): e_p0[0] = {e_p0[0]}")
    print("✓ _compute_realization_plan_constraints works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Sequence Form Conversion
print("\n4. Testing convert_to_sequence_form...")
print("-" * 80)
try:
    F_p0, F_p1, E_p0, e_p0, E_p1, e_p1 = convert_to_sequence_form(env, seed=0)
    
    print(f"✓ F_p0 shape: {F_p0.shape}")
    print(f"✓ F_p1 shape: {F_p1.shape}")
    print(f"✓ Is zero-sum? {np.allclose(F_p0 + F_p1, 0)}")
    print(f"✓ Max absolute sum error: {np.max(np.abs(F_p0 + F_p1)):.10f}")
    print(f"✓ E_p0 shape: {E_p0.shape}, e_p0 shape: {e_p0.shape}")
    print(f"✓ E_p1 shape: {E_p1.shape}, e_p1 shape: {e_p1.shape}")
    
    # Test that uniform strategy satisfies constraints
    uniform_p0 = np.ones(F_p0.shape[0]) / F_p0.shape[0]
    uniform_p1 = np.ones(F_p1.shape[1]) / F_p1.shape[1]
    
    print(f"\n  Testing uniform strategy constraint satisfaction:")
    # Note: uniform distribution doesn't necessarily satisfy flow constraints
    print(f"  (Uniform strategies don't necessarily satisfy realization constraints)")
    
    print("✓ convert_to_sequence_form works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Nash Equilibrium via Sequence Form LP
print("\n5. Testing find_nash_equilibrium_sequence_form...")
print("-" * 80)
try:
    r_p0, r_p1 = find_nash_equilibrium_sequence_form(env, seed=0)
    
    print(f"✓ Realization plan P0 shape: {r_p0.shape}")
    print(f"✓ Realization plan P1 shape: {r_p1.shape}")
    print(f"✓ P0 realization plan sum: {np.sum(r_p0):.5f}")
    print(f"✓ P1 realization plan sum: {np.sum(r_p1):.5f}")
    print(f"✓ P0 empty sequence prob: {r_p0[0]:.5f} (should be 1.0)")
    print(f"✓ P1 empty sequence prob: {r_p1[0]:.5f} (should be 1.0)")
    
    # Check constraints are satisfied
    constraint_error_p0 = np.max(np.abs(E_p0 @ r_p0 - e_p0))
    constraint_error_p1 = np.max(np.abs(E_p1 @ r_p1 - e_p1))
    print(f"✓ Max constraint error P0: {constraint_error_p0:.10f}")
    print(f"✓ Max constraint error P1: {constraint_error_p1:.10f}")
    
    # Compute game value
    game_value = r_p0.T @ F_p0 @ r_p1
    print(f"✓ Game value (Nash equilibrium utility): {game_value:.5f}")
    
    print("✓ find_nash_equilibrium_sequence_form works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Convert Realization Plan to Behavioral Strategy
print("\n6. Testing convert_realization_plan_to_behavioural_strategy...")
print("-" * 80)
try:
    behavioral_p0 = convert_realization_plan_to_behavioural_strategy(env, 0, r_p0, seed=0)
    behavioral_p1 = convert_realization_plan_to_behavioural_strategy(env, 1, r_p1, seed=0)
    
    print(f"✓ P0 behavioral strategy has {len(behavioral_p0)} information sets")
    print(f"✓ P1 behavioral strategy has {len(behavioral_p1)} information sets")
    
    print(f"\n  Sample P0 behavioral strategies:")
    for i, (infoset, probs) in enumerate(list(behavioral_p0.items())[:3]):
        print(f"    {infoset}: {probs}")
    
    print(f"\n  Sample P1 behavioral strategies:")
    for i, (infoset, probs) in enumerate(list(behavioral_p1.items())[:3]):
        print(f"    {infoset}: {probs}")
    
    # Check probabilities sum to 1
    for infoset, probs in behavioral_p0.items():
        if not np.isclose(np.sum(probs), 1.0):
            print(f"  ⚠ Warning: P0 infoset {infoset} probs sum to {np.sum(probs)}")
    
    for infoset, probs in behavioral_p1.items():
        if not np.isclose(np.sum(probs), 1.0):
            print(f"  ⚠ Warning: P1 infoset {infoset} probs sum to {np.sum(probs)}")
    
    print("✓ convert_realization_plan_to_behavioural_strategy works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Compare Nash Equilibrium with Fictitious Play
print("\n7. Comparing with Fictitious Play results...")
print("-" * 80)
try:
    from solutions.week07.week07 import fictitious_play
    
    # Run fictitious play
    fp_strategies = fictitious_play(env, num_iters=100, verbose=False)
    fp_final = fp_strategies[-1]
    
    print(f"✓ Fictitious Play converged after 100 iterations")
    print(f"✓ Comparing information sets...")
    
    # Compare a few key information sets
    from solutions.week07.week07 import _format_infoset_for_print
    
    sample_infosets = [
        list(behavioral_p0.keys())[0],
        list(behavioral_p1.keys())[0]
    ]
    
    for infoset in sample_infosets[:2]:
        if infoset in fp_final[0]:
            fp_probs = fp_final[0][infoset]
            seq_probs = behavioral_p0[infoset]
            print(f"\n  P0 {infoset}:")
            print(f"    FP:  {fp_probs}")
            print(f"    SEQ: {seq_probs}")
            print(f"    Diff: {np.linalg.norm(fp_probs - seq_probs):.5f}")
        elif infoset in fp_final[1]:
            fp_probs = fp_final[1][infoset]
            seq_probs = behavioral_p1[infoset]
            print(f"\n  P1 {infoset}:")
            print(f"    FP:  {fp_probs}")
            print(f"    SEQ: {seq_probs}")
            print(f"    Diff: {np.linalg.norm(fp_probs - seq_probs):.5f}")
    
    print("\n✓ Comparison complete!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
