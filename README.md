# RobustX <img width="100" alt="portfolio_view" align="right" src="https://isla-lab.github.io/images/slider/slider-image.svg"> <img width="100" alt="portfolio_view" align="right" src="https://upload.wikimedia.org/wikipedia/commons/0/06/Imperial_College_London_new_logo.png"> 

![Alt text](demo/RobustX.gif) 


> The increasing use of machine learning models to aid decision-making in high-stakes industries like finance and healthcare demands explainability to build trust. Counterfactual Explanations (CEs) provide valuable insights into model predictions by showing how slight changes in input data could lead to different outcomes. A key aspect of CEs is their robustness, which ensures that the desired outcomes remain stable even with minor alterations to the input. Robustness is important since produced CEs should hold up in the future should the original model be altered or replaced.Despite the importance of robustness, there has been a lack of standardised tools to comprehensively evaluate and compare robust CE generation methods. To address this, **RobustX** was developed as an open-source Python library aimed at benchmarking the robustness of various CE methods. RobustX provides a systematic framework for generating, evaluating, and comparing CEs with a focus on robustness, enabling fair and effective benchmarking. The library is highly extensible, allowing for custom models, datasets, and tools to be integrated, making it an essential tool for enhancing the reliability and interpretability of machine learning models in critical applications.

## Features

<table>
  <tr>
    <td><img src="demo/overview_new.png" width="850"></td>
    <td>
      <ul>
        <li> Standardises the evaluation and benchmarking of robust CEs.</li>
        <li> Supports multiple ML frameworks, including PyTorch, Keras, and scikit-learn.</li>
        <li>Extensible to incorporate custom models, datasets, CE methods, and evaluation metrics.</li>
        <li>Includes several robust CE generation algorithms (e.g., TRexNN, RNCE) and non-robust baselines (e.g., MCE, BLS).</li>      
      </ul>
    </td>
  </tr>
</table>


## Setup

The core required packages for working on RobustX are listed in ```environment.yml``` and ```requirements.txt```.

To set up a new virtual environment with Conda, use ```environment.yml```. This will install the required packages into a new environment called RobustX.

```bash
conda env create -f environment.yml
conda activate robustx
```

Alternatively, run ```pip install -r requirements.txt``` in a Python virtual environment if using an existing one. 

Note that one needs [Gurobi](https://www.gurobi.com/) optimizer to run mixed integer programming-based methods. Gurobi offers [free academic licenses](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

## Generating docs

Navigate to docs/source and run ```make html```. If that doesn't work, try to run ```sphinx-build -b html . _build/html```?

## Examples

A first example of RobustX is provided below.

```python
# first prepare a task
from robustx.datasets.ExampleDatasets import get_example_dataset
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.lib.tasks.ClassificationTask import ClassificationTask

data = get_example_dataset("ionosphere")
data.default_preprocess()
model = SimpleNNModel(34, [8], 1)
model.train(data.X, data.y)
task = ClassificationTask(model, data)

# then specify some CE methods
from robustx.generators.robust_CE_methods import MCER, RNCE, STCE
from robustx.generators.CE_methods import KDTreeNNCE, MCE
methods = {"MCER": MCER, "RNCE": RNCE, "STCE":STCE, "KDTreeNNCE": KDTreeNNCE, "MCE": MCE}

# benchmark the selected CE methods
from robustx.lib.DefaultBenchmark import default_benchmark
default_benchmark(task, methods, neg_value=0, column_name="target", delta=0.005)
```
which will produce an output similar to:

| Method     | Execution Time (s) | Validity proportion | Average Distance | Robustness proportion |
|------------|------------------:|--------------------:|-----------------:|----------------------:|
| MCER       |          92.6725  |                   1 |          5.09594 |              0.270833 |
| RNCE       |           2.25211 |                   1 |          6.3207  |              1        |
| STCE       |           9.74055 |                   1 |          6.86229 |              1        |
| KDTreeNNCE |           0.0779453|                   1 |          6.07593 |              0.416667 |
| MCE        |           1.65993 |                   1 |          2.45932 |              0        |




A demonstration of how to use the library is available here:

```bash
conda activate robustx
cd RobustX/demo
streamlit run demo_main.py --theme.base="light"   
```

Python notebooks demonstrating the usage of RobustX are
available [here](https://github.com/RobustX/RobustX/tree/main/examples).

The docs pages can be accessed by opening ```docs/build/html/index.html```.

## Contributors
* **Junqi Jiang** - junqi.jiang20@imperial.ac.uk
* **Luca Marzari** - luca.marzari@univr.it
* **Aaryan Purohit** 
* **Francesco Leofante** - f.leofante@imperial.ac.uk


## License

RobustX is licensed under the MIT License. For more details, please refer to the `LICENSE` file in this repository.
