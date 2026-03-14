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

The **`data`** folder is absent from the git repository due to high memory requirements and has to be taken from the published repository. 
Additionally, NOTAM data was not allowed for publishing. In order to setup a working project directory, the NOTAM data has to be downloaded manually in the format specified below in the shown project structure.
The project must include a **`data`** folder in its root directory with a specific structure as seen below.
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
    
### Notes:
- NOTAM files were not eligible for pusblishing. Therefore, the NOTAM files have to be downloaded manually into the correct folder with the correct structure for the project to work.
- Ensure all paths in the code are relative to the project root.


## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab (if using notebooks)
- Required Python packages (see `requirements.txt`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
