# Master Thesis: Assessing the Reliability of Flight Delay Predictions Using the Perturbation Approach

**Author:** Christoph Großauer

**Supervisor:** Assoz.-Prof. Mag. Dr. Christoph Schütz

**University:** Johannes Kepler Universität

**Institute:** Institute of Business Informatics - Data & Knowledge Engineering

---

## Project Overview
This project implements the Reliability Assessment Process via the Perturbation Approach within the aviation domain on a real world dataset.
For this, three models are trained to classify "early", "on-time" or "late" arrivals at Hartsfield-Jackson Airport in Atlanta Georgia United States.
After training and evaluation criteria are met for each model, the trained models' outputs are tested for reliability according to the Realiability Assessment Process with the Perturbation Approach.
Finally, these findings are analyzed and used as the basis for the research question of the connected Master's Thesis.


---

## Project Structure
The contents of the project looks as follows:

project_root/

├── data/                                 # Data folder

├── docs/                                 # Folder to store documentation and images

├── main/                                 # Includes all project files with python code

└── resources/                            # Includes shell scripts and python environment declaration

The path structure and filenames in this project must stay as shown in this repository for the relative paths to work as intended.

### Data Folder
The **`data`** folder holds all input data, trained models, processed files and perturbation matrices relevant to the project.
The **`data`** folder is absent from the git repository due to high memory requirements and has to be taken from the published repository. 
Additionally, NOTAM data was not allowed for publishing. In order to setup a working project directory, the NOTAM data has to be downloaded manually in the format specified below in the shown project structure.
The **`data`** folder must follow the project structure:

data/

├── input/

│   ├── data_raw/

│   │   ├── METAR_US/                 # METAR reports

│   │   ├── notams/

│   │   │   ├── katl/                  # KATL specific NOTAMs

│   │   │   │   ├── 01/                # NOTAMs captured in January

│   │   │   │   ├── 02/                # NOTAMs captured in February

│   │   │   │   ├── ...

│   │   │   │   └── 12/                # NOTAMs captured in December

│   │   ├── US_DomesticFlights/        # Flight data

│   │   │   ├── 2016/

│   │   │   ├── 2017/

│   │   │   └── ...

│   │   ├── airports.csv              # Airport data

│   │   ├── all_aircrafts_FAA.csv     # FAA aircraft data

│   │   └── runways.csv               # Runway data

│   └── scraped_aircraft/            # Scraped aircraft files

├── perturbation/

│   └── pert_output/                  # Perturbation outputs

├── preparation/

│   └── prepped_files/                # Prepared datasets

├── preprocessing/

│   └── base/

│       ├── class/                    # Test/Train split (classified target label)

│       ├── reg/                      # Test/Train split (continual target feature)

│       └── data.pkl                  # Integrated dataset (no split)

└── training/

└── training_results/             # Results of all trained models

### Main Folder
The **`main`** folder includes all project code for preprocessing steps, training processed, perturbation processes and analysis. 
The **`main`** folder is separated into the four big modules **`perturbation`**, **`preparation`**, **`preprocessing`** and **`training`**. Additionally, **`utils`** includes auxiliary functions that were potentially reused. The following subsections in this markdown file show the order in which the models are relevant to the project and show important files for each submodule.

#### Preparation Module ####
The **`preparation`** module is the initial point of contact to the project. By running the files as shown below in sequential order, all relevant data is loaded, converted to pickle files and slightly processed in order to be representable by a single data matrix with no null values. The last part of the module converts all units into readable scales and alters their naming to improve readability.

preparation/

├── 01_flight_prep.ipynb          # Load and prepare flight data

├── 02_airport_prep.ipynb          # Load and prepare airport data

├── 03_runway_prep.ipynb          # Load and prepare runway data

├── 04_aircraft_prep.ipynb          # Load and prepare aircraft data

├── 05_metar_prep.ipynb          # Load and prepare metar data

├── 06_notam_prep.ipynb          # Load and prepare metar data

├── 07_integration.ipynb          # Integrates all prepared datasets into a single data table

└── 08_name_value_conversion.ipynb  # Converts the files units and names

#### Preprocessing Module ####
The **`preprocessing`** module takes the final dataset established in the previous module and creates the classes for the target variable as well as splits data into test and train dataset. Aditionally, multiple data variations were created which are produced in the corresponding **`production`** folder. Therefore the **`CCLASS`** dataset that is often mentioned, stems from the "C" variation of input data and has the target feature represented within the target classes. The variations are created based on experimentations and findings in analysis files.

├── preprocessing/

├── analysis/          # Analysing specific aspect of the data files

├── production/          # Creating the varying data variations

├── 01_target_creation.ipynb          # Creates the target classes for the combined dataset

└── 02_test_train_split.ipynb          # Split data into training and test set

#### Training Module ####
The **`training`** module trains a model based on parameters given to the **`train.py`** file. This file is the main file used for training and uses the **`estimation.py`** and **`data.py`** files as auxiliary objects to handle estimation operations and data loading respectively. The **`estimation.py`** file includes the dictionaries for parameter estimation hardcoded. The notebook file **`baseline_training.ipynb`** created the baselines for the target metrics. Finally, **`03_evaluation_2.0.ipynb`** shows the evaluation of the models and baselines in confusion matrices.

training/

├── analysis/

│   └── 03_evaluation_2.0.ipynb                 # Evaluationg trained models and baselines with confusion matrices

├── baseline_training.ipynb         # Constructs baseline models

├── data.py         # Data holder object for the training process

├── estimation.py         # Estimator object responsable for parameter estimation settings

└── train.py         # Trains a model based on the given parameters (main function for training)


#### Perturbation Module
The **`perturbation`** module loads the trained models and their respective test data from which it creates the perturbed input matrix that is given to the loaded model to evaluate the reliability of the predictions.
The python file **`01_perturbation_run.py`** creates the perturbation dataset that is analysed and altered with additional rows for better readability and identification of perturbation findings in **`02_perturbation_analysis_{model_name}.ipynb`**. The latter creates the **`pert_view_{model_name}.pkl`** and **`pert_test_results_{model_name}.pkl`** files that are used for graphical and quantitative analysis of the perturbation results respectively.

perturbation/                    # Includes all used perturbation notebooks and the python file to create perturbed data entries     

├── analysis/                    # Additional analysis regarding perturbation

├── saved_files/                 # Jupyter notebooks saved as HTML to allow looking up old versions

├── 01_perturbation_run.py       # Python file that constructs the perturbed data inputs for further reliability analysis and stores them as .pkl

├── 02_perturbation_analysis_{model_name}.ipynb       # Jupyter notebooks for perturbation analysis for each model

├── pert_view_{model_name}.pkl # Created data matrix for graphical analysis

└── pert_test_results_{model_name}.pkl # Created data matrix for quantitative analysis

### Notes
- All files in analysis folders are created to gather knowledge of the dataset and formulate decisions based on these findings as described in the text of the thesis.
- All files in archive folders are not functional and only retained for documentation purposes.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab (if using notebooks)
- Required Python packages (see `requirements.txt`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
