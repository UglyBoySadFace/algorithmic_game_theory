#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.week08.week08 import convert_to_normal_form
import numpy as np

env = KuhnPoker()
p0_payoff, p1_payoff = convert_to_normal_form(env, seed=0)

print(f'Matrix dimensions: {p0_payoff.shape}')
print(f'Number of strategies for P0: {p0_payoff.shape[0]}')
print(f'Number of strategies for P1: {p0_payoff.shape[1]}')
print(f'Is zero-sum? {np.allclose(p0_payoff + p1_payoff, 0)}')
print(f'Max absolute sum error: {np.max(np.abs(p0_payoff + p1_payoff))}')
print()
print(f'Sample payoffs (first 5x5):')
print(f'P0 payoffs:')
print(p0_payoff[:5, :5])
print(f'\nP1 payoffs:')
print(p1_payoff[:5, :5])
print(f'\nSum (should be all zeros):')
print(p0_payoff[:5, :5] + p1_payoff[:5, :5])
print()
print(f'Min P0 payoff: {np.min(p0_payoff):.4f}')
print(f'Max P0 payoff: {np.max(p0_payoff):.4f}')
print(f'Mean P0 payoff: {np.mean(p0_payoff):.4f}')
