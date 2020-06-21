Analysis of AI research to tackle COVID-19
==============================

This code contains scripts and notebooks to:

1. Reproduce the analysis presented in [Nesta's report about AI and the fight against COVID-19](https://www.nesta.org.uk/report/artificial-intelligence-and-fight-against-covid-19/) (AI-C19)
2. Update data collections (this requires access to Nesta's data production system)
3. Reproduce future analyses based on updated data collections

## Instructions

### Setup
1. Create a conda virtual environment with the packages we use in our analysis:

`conda env create -f conda_environment.yaml`

2. Install scripts as a package:
`pip install -e .`

3. If you have access to Nesta DAPS and are planning to access data from there, install the `data_getters` package:
`pip install -r nesta_packages.txt`

### Collect data
You can collect the processed data we used in the AI-C19 report from [figshare](https://figshare.com/articles/Artificial_Intelligence_and_the_Fight_Against_COVID-19/12479570) by running:

`python ai_covid_19/make_dataset.py`

The downloaded files also include data dictionaries.

You can make a new dataset with (probably updated data) by running:

`python ai_covid_19/fetch_data.py`

**Note:** This requires putting your credentials in a `.env` file that will be read by the relevant scripts

### Analysis
Each notebooks in the `notebooks/ai-c19` folder refers to a section in the paper.

You can re-run them individually. All visual outputs will be saved as html files in the `report/figures/ai_c19` folder.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/nestauk/cookiecutter-data-science-nesta">Nesta cookiecutter data science project template</a>.</small></p>
