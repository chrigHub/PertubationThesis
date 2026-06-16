# Master Thesis: Assessing the Reliability of Flight Delay Predictions Using the Perturbation Approach

**Author:** Christoph Großauer

**University:** Johannes Kepler Universität Linz

**Institute:** Institute of Business Informatics - Data & Knowledge Engineering

---

## Project Overview
This project implements the Reliability Assessment Process in combination with the Perturbation Approach within the aviation domain on a real world dataset.
For this, three models are trained to classify "early", "on-time" or "late" arrivals at Hartsfield-Jackson Airport in Atlanta, Georgia, United States.
After training and evaluation criteria are met for each model, the trained models' outputs are tested for reliability according to the Reliability Assessment Process with the Perturbation Approach.
Finally, these findings are analyzed and used as the basis for the research question of the connected Master's Thesis.


---

## Project Structure
The contents of the project look as follows:

project_root/

├── data/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    # Data folder

├── main/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                                # Includes all project files with python code

└── resources/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                           # Includes shell scripts and python environment declaration

The path structure and filenames in this project must stay as shown in this example for the relative paths to work as intended.

### Data Folder
The **`data`** folder holds all input data, trained models, processed files and perturbation matrices relevant to the project.
The **`data`** folder is absent from the git repository due to high memory requirements and has to be taken from the published repository. 
Additionally, NOTAM data was not allowed for publishing. In order to set up a working project directory, the NOTAM data has to be downloaded manually in the format specified below in the shown project structure.
The **`data`** folder must follow the project structure:

data/

├── input/

│   ├── data_raw/

│   │   ├── METAR_US/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                # METAR reports

│   │   ├── notams/

│   │   │   ├── katl/   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               # KATL specific NOTAMs

│   │   │   │   ├── 01/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            # NOTAMs captured in January

│   │   │   │   ├── 02/   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             # NOTAMs captured in February

│   │   │   │   ├── ...

│   │   │   │   └── 12/   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             # NOTAMs captured in December

│   │   ├── US_DomesticFlights/  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Flight data

│   │   │   ├── 2016/

│   │   │   ├── 2017/

│   │   │   └── ...

│   │   ├── airports.csv    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Airport data

│   │   ├── all_aircrafts_FAA.csv  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   # FAA aircraft data

│   │   └── runways.csv     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Runway data

│   └── scraped_aircraft/     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Scraped aircraft files

├── preparation/

│   └── prepped_files/      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Prepared datasets

├── preprocessing/

│   └── base/

│       ├── class/           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         # Test/Train split (classified target label)

│       ├── reg/            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Test/Train split (continual target feature)

│       └── data.pkl       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           # Integrated dataset (no split)

└── training/

 └── training_results/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         # Results of all trained models

### Main Folder
The **`main`** folder includes all project code for preprocessing steps, training processes, perturbation processes and analysis. 
The **`main`** folder is separated into the four big modules **`perturbation`**, **`preparation`**, **`preprocessing`** and **`training`**. Additionally, **`utils`** includes auxiliary functions that were potentially reused. The following subsections in this markdown file show the order in which the models are relevant to the project and show important files for each submodule.

#### Preparation Module ####
The **`preparation`** module is the initial point of contact to the project. 
By running the files as shown below in sequential order, all relevant data is loaded, converted to pickle files and
slightly processed in order to be representable by a single data matrix with no null values. The last part of the module
converts all units into readable scales and alters their naming to improve readability.

preparation/

├── 01_flight_prep.ipynb    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Load and prepare flight data

├── 02_airport_prep.ipynb    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Load and prepare airport data

├── 03_runway_prep.ipynb     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Load and prepare runway data

├── 04_aircraft_prep.ipynb     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Load and prepare aircraft data

├── 05_metar_prep.ipynb     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Load and prepare METAR data

├── 06_notam_prep.ipynb   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Load and prepare NOTAM data

├── 07_integration.ipynb   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Integrates all prepared datasets into a single data table

└── 08_name_value_conversion.ipynb &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Converts the features' units and names

#### Preprocessing Module ####
The **`preprocessing`** module takes the final dataset established in the previous module and creates the classes for
the target variable as well as splits data into test and train datasets. Additionally, multiple data variations were
created which are produced in the corresponding **`production`** folder. Therefore, the **`CCLASS`** dataset that is
often mentioned stems from the "C" variation of input data and has the target feature represented within the target
classes. The variations are created based on experimentation and findings in analysis files.

├── preprocessing/

├── analysis/   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Analysing specific aspects of the data file

├── production/   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Create the varying data variations

├── 01_target_creation.ipynb      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    # Create the target classes for the combined dataset

└── 02_test_train_split.ipynb    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Splits data into training and test set

#### Training Module ####
The **`training`** module trains a model based on parameters given to the **`train.py`** file. This file is the main
file used for training and uses the **`estimation.py`** and **`data.py`** files as auxiliary objects to handle
estimation operations and data loading respectively. The **`estimation.py`** file includes the dictionaries for
parameter estimation hardcoded. The notebook file **`baseline_training.ipynb`** creates the baselines for the target
metrics. Finally, **`03_evaluation_2.0.ipynb`** shows the evaluation of the models and baselines in confusion matrices.

training/

├── analysis/

│   └── 03_evaluation_2.0.ipynb    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             # Evaluating trained models and baselines with confusion matrices

├── baseline_training.ipynb  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Constructs baseline models

├── data.py   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Data holder object for the training process

├── estimation.py  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # Estimator object responsible for parameter estimation settings

└── train.py   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Trains a model based on the given parameters (main function for training)


#### Perturbation Module ####
The **`perturbation`** module loads the trained models and their respective test data from which it creates the
perturbed input matrix that is given to the loaded model to evaluate the reliability of the predictions.
The Python file **`01_perturbation_run.py`** creates the perturbation dataset that is analysed and altered with
additional rows for better readability and identification of perturbation findings in
**`02_perturbation_analysis_{model_name}.ipynb`**. The latter creates the **`pert_view_{model_name}.pkl`** and
**`pert_test_results_{model_name}.pkl`** files that are used for graphical and quantitative analysis of the
perturbation results, respectively.

perturbation/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                # Includes all used perturbation notebooks and the python file to create perturbed data entries     

├── analysis/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                # Additional analysis regarding perturbation

├── 01_perturbation_run.py   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    # Python file that constructs the perturbed data inputs for further reliability analysis and stores them as .pkl

├── 02_perturbation_analysis_{model_name}.ipynb  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Jupyter notebooks for perturbation analysis for each model

├── pert_view_{model_name}.pkl &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  # Created data matrix for graphical analysis

└── pert_test_results_{model_name}.pkl &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  # Created data matrix for quantitative analysis

### Resources Folder ###
The resources folder contains shell scripts that were used to call the trai.py file with the correct parameters as well
as the conda environment yaml file and connected script files.

resources/                   

├── env/      

│   ├── ma-env.yml      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           # YAML file with needed dependencies for the project

│   └── sync_env.sh    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;             # Shell script to load the YAML file dependencies

└── sh/       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Folder with shell scripts to run the train.py file

### Notes
- All files in analysis folders are created to gather knowledge of the dataset and formulate decisions based on these findings as described in the text of the thesis.

## Project Run Instructions

### 1. Prerequisites
- Python 3.8+
- Conda 22.9+
- Jupyter Notebook/Lab
- Required Python packages as given by the conda environment
- ./data/input/ file structure as seen in the **`Data Folder`** section

### 2. Project steps
In order to simulate a full run of all project steps with a newly trained model based on the parameters set on various scripts and estimation settings as defined in the estimation.py file, follow the steps below.

1. Run all files in ./main/preparation/ in the order given by their filename
2. Run all files in ./main/preprocessing/ in the order given by their filename
3. Run the file in the ./main/preprocessing/production/ folder as needed
4. Run the ./main/training/train.py file with fitting parameters. For example as seen in ./resoures/sh/
5. Run the ./main/perturbation/perturbation_run.py file with fitting model and folder parameter name to indicate the model type and version. For example  "--model XGB --folder 2024_06_17-1513"
6. Alter the "loading_folder" and "loading_spec" variables in the first cell of ./main/perturbation/02_perturbation_analysis.ipynb and run the file
