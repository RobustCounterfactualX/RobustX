# RobustX

The increasing use of machine learning models to aid decision-making in high-stakes industries like finance and
healthcare demands explainability to build trust. Counterfactual Explanations (CEs) provide valuable insights into model
predictions by showing how slight changes in input data could lead to different outcomes. A key aspect of CEs is their
robustness, which ensures that the desired outcomes remain stable even with minor alterations to the input. Robustness
is important since produced CEs should hold up in the future should the original model be altered or replaced.Despite
the importance of robustness, there has been a lack of standardised tools to comprehensively evaluate and compare robust
CE generation methods. To address this, **RobustX** was developed as an open-source Python library aimed at benchmarking
the robustness of various CE methods. RobustX provides a systematic framework for generating, evaluating, and comparing
CEs with a focus on robustness, enabling fair and effective benchmarking. The library is highly extensible, allowing for
custom models, datasets, and tools to be integrated, making it an essential tool for enhancing the reliability and
interpretability of machine learning models in critical applications.

## Features

- Standardises the evaluation and benchmarking of robust CEs.
- Supports multiple ML frameworks, including PyTorch, Keras, and scikit-learn.
- Extensible to incorporate custom models, datasets, CE methods, and evaluation metrics.
- Includes several robust CE generation algorithms (e.g., TRexNN, RNCE) and non-robust baselines (e.g., MCE, BLS).

## Setup

The core required packages for working on RobustX are listed in ```environment.yml``` and ```requirements.txt```.

To set up a new virtual environment with Conda, use ```environment.yml```. This will install the required packages into a new environment called RobustX.

```bash
conda env create -f environment.yml
conda activate robustx
```

Alternatively, run ```pip install -r requirements.txt``` in a Python virtual environment if using an existing one. 

Note that one needs [Gurobi](https://www.gurobi.com/) optimizer to run mixed integer programming based methods. Gurobi offers [free academic licenses](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

## Generating docs

Navigate to docs/source and run ```make html```. If that doesn't work, try to run ```sphinx-build -b html . _build/html```?

## Examples

Python notebooks demonstrating the usage of RobustX are
available [here](https://github.com/RobustX/RobustX/tree/main/examples).

The docs pages can be accessed by opening ```docs/build/html/index.html```.

For a step-by-step tutorial, you can also watch the video guide [here](https://youtu.be/z9pbCFg9xVA?si=MjgZPDVBMumQ7ccu)
.

## License

RobustX is licensed under the MIT License. For more details, please refer to the `LICENSE` file in this repository.
