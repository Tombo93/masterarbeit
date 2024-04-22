Masterarbeit
==============================

Install
------------
    $ make requirements
    
Set up project & download data
------------
    $ make data

Run experiments
------------
    $ make isic



Project Organization
------------

    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── data
    │   ├── interim
    |   |   ├── cifar10      
    │   |   └── isic  
    │   ├── processed
    │   |   ├── cifar10      
    │   |   └── isic            
    │   └── raw      
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models
    │
    ├── notebooks
    │ 
    │ 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports
    |   ├── cifar10      
    │   ├── figures
    |   └── isic      
    │
    ├── encironment.yml
    │                      
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization
    │       ├── plot_cifar.py
    │       ├── plot_isic.py
    │       └── plot_confusion_matrix.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
