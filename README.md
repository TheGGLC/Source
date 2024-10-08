This repository includes the source code of [TheGGLC dataset](https://github.com/TheGGLC/TheGGLC).

## Quick Environment Setup
`pipenv install`

`pipenv shell`

## Create Solvable Levels
`./playable.sh <game> <root_folder>`
## Create Unsolvable Levels
`./unplayable.sh <game> <root_folder>`
# Experiment Section Reproducibility
## Create Embeddings
`git clone https://github.com/TheGGLC/TheGGLC.git`

`python ./analysis/clip.py --game <game>`

`python ./analysis/uumap.py --game <game>`
## Embeddings Space Visualization
`python ./analysis/visualize.py --game <game>`
## Astuteness Comparison
`python ./analysis/metric_compare.py`

# Other Tools
## The Solvability Determination Tool
`./solution.sh <game> <input_file_path> <output_file_path>`
