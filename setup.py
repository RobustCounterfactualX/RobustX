from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='robustx',
        version='1.0.0',
        packages=find_packages(),
        install_requires=[
            'torch==2.4.1',
            'gurobipy==11.0.1',
            'clingo>=5.6.2',
            'tensorflow==2.16.2',
            'keras==3.0.5',
            'pytest',
            'notebook',
            'numpy',
            'pandas',
            'scikit-learn',
            'scipy',
            'tqdm',
            'tabulate',
            'streamlit',
            'matplotlib',
            'seaborn',
        ],
        url="https://github.com/RobustCounterfactualX/RobustX",
        author='Junqi Jiang, Luca Marzari, Aaryan Purohit, Francesco Leofante',
        description='A Python library for benchmarking robust counterfactual explanations.',
        python_requires='>=3.9',
    )
