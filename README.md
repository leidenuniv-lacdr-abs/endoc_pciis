# Application of post-column infused standard (PCIS) correction of matrix effect
The code and data provided here will allow anyone to reproduce analyses and results described in:

> Anne-Charlotte Dubbelman, Bo van Wieringen, Lesley Roman Arias, Michael van Vliet, Roel Vermeulen, Amy C. Harms, Thomas Hankemeier. Strategies for using post-column infusion of standards to correct for matrix effect in LC-MS-based quantitative metabolomics. 2023

[Strategies for Using Postcolumn Infusion of Standards to Correct for Matrix Effect in LC-MS-Based Quantitative Metabolomics](https://pubs.acs.org/doi/10.1021/jasms.4c00408)
[https://doi.org/10.1021/jasms.4c00408](https://pubs.acs.org/doi/10.1021/jasms.4c00408)

## Run in the cloud
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/leidenuniv-lacdr-abs/endoc_pciis/HEAD)

## Local installation

Recommended is to
- install Python 3.x [Python 3.x](https://www.python.org/downloads/)
- download or clone this repository, and cd into the directory of the repository
- create and enable a virtual environment: [Creation of virtual environments](https://docs.python.org/3/library/venv.html)
- install the required libraries using the requirements.txt file: pip install -r requirements.txt
- start the notebook with: jupyter notebook [Launching Jupyter Notebook App](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html)

## Data

The demo data comprises the mzmL files of the LC-MS/MS injections that were used for the experiment decribed in Scheme 1, Method 1 as described in _Strategies for using post-column infusion of standards to correct for matrix effect in LC-MS-based quantitative metabolomics._ by Dubbelman _et al._ 

## Acknowledgement and Citations
As with many other scientific software, this would have been a lot more difficult to create without all the great open-source libraries out there. Please find all the libaries used in the requirements.txt file, and special thanks to:

> Marcos Duarte. (2021), Zenodo, detecta: A Python module to detect events in data (Version v0.0.5). Zenodo. http://doi.org/10.5281/zenodo.4598962

> The pandas development team (2020), Zenodo, Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool https://doi.org/10.5281/zenodo.3509134

> SciPy Contributors (2020), Nature Methods, Fundamental Algorithms for Scientific Computing in Python https://rdcu.be/b08Wh

> Pearson K. A., et al. (2019), The Astronomical Journal, Ground-based Spectroscopy of the Exoplanet XO-2b Using a Systematic Wavelength Calibration http://iopscience.iop.org/article/10.3847/1538-3881/aaf1ae/meta

> Jacob VanderPlas and Brian Granger and Jeffrey Heer and Dominik Moritz and Kanit Wongsuphasawat and Arvind Satyanarayan and Eitan Lees and Ilia Timofeev and Ben Welsh and Scott Sievert (2018), Journal of Open Source Software, Altair: Interactive Statistical Visualizations for Python https://doi.org/10.21105/joss.01057
