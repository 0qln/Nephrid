#!/usr/bin/env bash

# learn endgames
python resources/datasets/extract_phase.py --phase 0 --themes "mateIn1"
python resources/datasets/extract_phase.py --phase 1 --themes "mateIn1,mateIn2"
python resources/datasets/extract_phase.py --phase 2 --themes "mateIn1,mateIn2,mateIn3"
python resources/datasets/extract_phase.py --phase 3 --themes "mateIn2,mateIn3"
python resources/datasets/extract_phase.py --phase 4 --themes "mateIn3,mateIn4"
python resources/datasets/extract_phase.py --phase 5 --themes "mateIn4,mateIn5"

# learn middlegames
