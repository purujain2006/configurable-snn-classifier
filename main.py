#!/usr/bin/env python3
"""snnsearch entry point.

    python main.py check                                # what is missing here
    python main.py summary -c configs/dvs128.yaml       # no torch, no dataset
    python main.py single  -c configs/dvs128.yaml
    python main.py search  -c configs/cifar10.yaml
    python main.py report  results/dvs128

Everything else lives in the snnsearch package. This file stays a launcher so
there is one obvious thing to run.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snnsearch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
