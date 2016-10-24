#!/bin/sh
python classification.py
python classification_grid_search.py
python regression.py
python regression_grid_search.py
python find_best_round.py
python save_load_model.py