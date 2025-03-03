# RoCELib

The increasing use of machine learning models to aid decision-making in high-stakes industries like finance and
healthcare demands explainability to build trust. Counterfactual Explanations (CEs) provide valuable insights into model
predictions by showing how slight changes in input data could lead to different outcomes. A key aspect of CEs is their
robustness, which ensures that the desired outcomes remain stable even with minor alterations to the input. Robustness
is important since produced CEs should hold up in the future should the original model be altered or replaced.Despite
the importance of robustness, there has been a lack of standardised tools to comprehensively evaluate and compare robust
CE generation methods. To address this, **RoCELib** was developed as an open-source Python library aimed at benchmarking
the robustness of various CE methods. RoCELib provides a systematic framework for generating, evaluating, and comparing
CEs with a focus on robustness, enabling fair and effective benchmarking. The library is highly extensible, allowing for
custom models, datasets, and tools to be integrated, making it an essential tool for enhancing the reliability and
interpretability of machine learning models in critical applications.

## Features

- Standardises the evaluation and benchmarking of robust CEs.
- Supports multiple ML frameworks, including PyTorch, Keras, and scikit-learn.
- Extensible to incorporate custom models, datasets, CE methods, and evaluation metrics.
- Includes several robust CE generation algorithms (e.g., TRexNN, RNCE) and non-robust baselines (e.g., MCE, BLS).

## Setup

To set up do:

(1) Create a virtual environment in the root

```bash
python -m venv venv
```

(2) Activate it. (You may have to use a different command depending on your platform)

```bash
source venv/bin/activate
```

(3) Download dependencies:

```bash
pip install -r requirements.txt
```

If you want to also run tests, regenerate docs, do the below, this installs rocelib itself as library:

```pip install .```

You need Gurobi to run all tests.

If you are on PyCharm, to run tests make sure that your default tester is pytest. (Settings > Tools > Python Integrated Tools > Default Test Runner > pytest)

## Generating Docs

The documentation for RoCELib is automatically generated and hosted online. You can access the latest version here:

🔗 [RoCELib Documentation](https://jstadl.github.io/RoCELib/rocelib.html)

There is no need to manually build the documentation unless making local modifications. However, if you need to generate docs locally, navigate to `docs/source` and run:

```bash
make html
```

If that does not work, try:

```bash
sphinx-build -b html . _build/html
```

## Examples

Python notebooks demonstrating the usage of RoCELib are
available [here](https://github.com/RoCELib/RoCELib/tree/main/examples).

The docs pages can be accessed by opening ```docs/build/html/index.html```.

For a step-by-step tutorial, you can also watch the video guide [here](https://youtu.be/z9pbCFg9xVA?si=MjgZPDVBMumQ7ccu)
.

## Release Process

For detailed instructions on publishing a new version of RoCELib to PyPI, refer to the [RELEASE.md](RELEASE.md) file in the repository. This file contains step-by-step guidance on versioning, tagging, and triggering the GitHub Action workflow for automated publishing.

## License

RoCELib is licensed under the MIT License. For more details, please refer to the `LICENSE` file in this repository.
