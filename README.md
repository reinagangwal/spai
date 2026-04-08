# SPENDLY — Team Setup Instructions

Welcome to the **SPENDLY** project repository! This guide provides step-by-step instructions for all team members (Person 1, Person 2, Person 3, Person 4) to clone the repository, set up their Python environment, and start working with the shared dataset.

## 1. Clone the Repository
Start by cloning the shared Git repository to your local machine:
```bash
git clone https://github.com/reinagangwal/spai.git
cd spai
```
*(Note: Ensure you are cloning into an appropriate workspace directory.)*

## 2. Set Up a Virtual Environment
To avoid package conflicts, everyone must use an isolated virtual environment.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Required Packages
Once your virtual environment is active, install the shared package dependencies:
```bash
pip install -r requirements.txt
```
*Note for Person 4: The `requirements.txt` is updated to install `torch` by default instead of `tensorflow` due to Python 3.14 compatibility.*

## 4. Run the Data Preprocessing Pipeline
Person 1 has finalized the data preprocessing logic. To load, clean, engineer, scale, and split the raw Kaggle dataset, run:
```bash
python src/preprocess.py
```
This script will automatically create a `data/processed/` directory.

## 5. Working With the Data
All teammates can now independently import the clean, standardized CSV files generated in the previous step:
- `data/processed/train.csv` (Earliest 70%)
- `data/processed/val.csv` (Next 15%)
- `data/processed/test.csv` (Last 15%)

These splits contain exactly the encoded, standardized, and clean columns agreed upon in our schema contract (see `src/preprocess.py` comments for details).

---
**Happy coding, team! Please create feature branches before pushing your own changes (e.g., `git checkout -b your-feature`).**
