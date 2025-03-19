# About

Repository of files referenced in research by Gus Lathouwers, Wieke Harmsen, Catia Cucchiarini, Helmers Strik; Radboud University (2024).

Contains the three datasets used for syllabification benchmarking as well as two algorithms (neural net and Brandt). 

Also see https://github.com/guslatho/syllabificator for an older version of the research repository containing additional syllabification algorithms.

# Usage

'torch_train.ipynb' Is a sample setup for training a new torch model to perform syllabification. The torch training is language agnostic, meaning any language file can be input (currently only tested on languages using the latin alphabet).

'sample_script.ipynb' Contains reference code to use either the Brandt algorithm or an pretrained torch model for syllabification.

# Misc

### Version Info:
- `Torch`: `2.5.1`  
- `Python`: `3.12.7`  
- `BiLSTM CRF`: `0.2.1`  
