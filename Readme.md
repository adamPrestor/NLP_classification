# Book discussions message classification

Project contains the implementation of several models and feature extractors for message classification.

## Report

Report is available [here](report/NLP_Classification.pdf).

## Setup

1. Install the required packages

```bash
pip install -r requirements.txt
```

## Running the experiments

A simplified version of the experiments (using a single dataset split instead of cross validation) can be run using the following command. The script should finish in a minute or two.

```bash
python classification.py
```

To run the full experiment used to produce the plots and tables in the report, use the script with the `--full` flag.

```bash
python classification.py --full
```

The notebook `cross_validation.ipynb` contains the same script in notebook form and is a good starting point for playing around with custom models and feature sets.
