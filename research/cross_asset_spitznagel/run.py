#!/usr/bin/env python3
"""Orchestrator for the Cross-Asset Spitznagel research suite.

Runs one or more of the individual analysis scripts:
  equity     -- S&P 500 (ES) + tail hedge
  treasury   -- US Treasuries (ZN/ZB) + tail hedge
  bond_carry -- US-UK bond carry (ZN vs Gilt) + directional hedge
  commodity  -- Commodity carry (GC/CL/HG/NG) + OTM puts
  combined   -- Multi-strategy portfolio combining ES + FX + Bond Carry

Usage:
  python run.py                   # run all five
  python run.py equity treasury   # run only equity and treasury
  python run.py combined          # run only the combined portfolio
"""
import argparse
import sys
import time


MODULES = {
    'equity':     'run_equity',
    'treasury':   'run_treasury',
    'bond_carry': 'run_bond_carry',
    'commodity':  'run_commodity',
    'combined':   'run_combined',
}


def main():
    parser = argparse.ArgumentParser(
        description='Run cross-asset Spitznagel tail-hedge analyses.',
    )
    parser.add_argument(
        'modules', nargs='*', default=list(MODULES.keys()),
        choices=list(MODULES.keys()) + [[]],
        help='Which analyses to run (default: all)',
    )
    args = parser.parse_args()

    targets = args.modules if args.modules else list(MODULES.keys())

    for name in targets:
        mod_name = MODULES[name]
        print(f'\n{"=" * 70}')
        print(f'  RUNNING: {name}  ({mod_name}.py)')
        print(f'{"=" * 70}\n')

        t0 = time.time()
        try:
            mod = __import__(mod_name)
            mod.main()
        except Exception as exc:
            print(f'\nERROR in {name}: {exc}', file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - t0
        print(f'\n--- {name} finished in {elapsed:.1f}s ---')

    print('\nAll done.')


if __name__ == '__main__':
    main()
