from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='robustx',
        version='0.1.0',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            'scikit-learn',
        ],
        author='Junqi Jiang, Luca Marzari, Aaryan Purohit, Francesco Leofante',
        description='A library for generating robust counterfactual explanations.',
        python_requires='>=3.7',
    )
