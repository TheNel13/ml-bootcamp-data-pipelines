# ML Bootcamp – Data Preparation Pipelines

This project builds two reusable data preparation pipelines using real-world datasets. The goal is to practice framing data science questions, defining business metrics, and constructing end-to-end data preparation workflows that output model-ready train, tune, and test datasets.

Each pipeline is implemented as a sequence of reusable functions (a data prep DAG) rather than a single monolithic script.

---

## Repository Structure

ML-BOOTCAMP-DATA-PIPELINES/
├── college_pipeline.py
├── placement_pipeline.py
├── README.md
├── .gitignore
└── data/
├── cpipline/
│ ├── cc_institution_details.csv
│ ├── cc_institution_grads.csv
│ ├── cc_state_sector_details.csv
│ └── cc_state_sector_grads.csv
└── ppipeline/
└── Placement_Data_Full_Class.csv

---

## Dataset 1: College Completion

**Dataset:**  
IPEDS-based institutional college completion data (`cc_institution_details.csv`).

### Problem Question
Can institutional characteristics be used to predict whether a college has a high student completion rate?

### Business Metric / Target
High vs. low completion, defined using the IPEDS-standard **150% graduation rate** (`grad_150_value`).  
Institutions with a 150% graduation rate greater than or equal to 0.5 are labeled as high completion.

### Data Preparation Steps
- Created a binary target variable from the 150% graduation rate
- Removed identifier columns (institution name and unit ID)
- One-hot encoded categorical variables
- Normalized continuous variables using standard scaling
- Calculated target prevalence
- Split the data into Train / Tune / Test partitions (70 / 15 / 15)

### Instincts & Concerns
One-hot encoding high-cardinality categorical variables (such as institution names and locations) significantly increases feature dimensionality, which may introduce sparsity and overfitting risk. Additionally, because the data is institution-level rather than student-level, it may not capture important individual factors influencing completion outcomes.

---

## Dataset 2: Job Placement

**Dataset:**  
Campus placement data (`Placement_Data_Full_Class.csv`).

### Problem Question
Can student academic and demographic features be used to predict whether a student will be placed in a job?

### Business Metric / Target
Job placement status (Placed vs. Not Placed).

### Data Preparation Steps
- Created a binary placement target from the placement status column
- Removed identifier and leakage variables (student ID and salary)
- One-hot encoded categorical variables
- Normalized continuous variables using standard scaling
- Calculated target prevalence
- Split the data into Train / Tune / Test partitions (70 / 15 / 15)

### Instincts & Concerns
The dataset is relatively small, which increases the risk of overfitting. While academic performance variables are informative, important employability factors such as interview performance and networking are not captured in the data.

---

## Environment Setup

A Python virtual environment was used to manage dependencies.

Required packages:
- pandas
- scikit-learn

---

## How to Run

Activate the virtual environment and run the pipelines from a Python shell:

```python
from college_pipeline import college_pipeline
college_pipeline()

from placement_pipeline import placement_pipeline
placement_pipeline()