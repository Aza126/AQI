# AQI
Air Quality Index Dashboard - for an assignment

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)

##### Steps for training models

###### B1: Clone and create virtual environment
```bash
git clone git@github:Aza126/AQI.git
# 'yes', spam 'Enter'
cd AQI
python -m venv .venv
mkdir -p artifacts models/rf models/lstm
```
###### B2: Activate virtual environment
```bash
# Linux/MacOS
source .venv/bin/activate
# Windows
.venv/Scripts/activate
```
###### B3: Create .env -> ask Aza for the words needed
```bash
cp .env.example .env
# or
copy .env.example .env
# replace some words in .env
```
###### B4: Download needed librabies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
###### B5: Download needed files
```bash
python -m src.scripts.download_models
```
###### B6: Train your models
```bash
# Write and run your `rf.py` and `lstm.py`
# using data from `scaler.pkl` and `training_data_scaled.parquet`
# to create models `rf_v1.pkl` and `lstm_v1.h5`
```
###### B7: Push your code to remote repo using other branches
```bash
git branch feature/rf feature/lstm

git switch feature/rf
git add src/training/rf.py
git commit -m "Update rf.py"
git push -u origin feature/rf

git switch feature/lstm
git add src/training/lstm.py
git commit -m "Update lstm.py"
git push -u origin feature/lstm
```
###### B8: Save your `rf.pkl` and `lstm.h5` to Google Drive
```bash
# Replace the old empty ones
```
###### If you wanna run dashboard
```bash
python -m src.inference
python -m streamlit run dashboard/app.py
# How to stop
Ctrl + C 
```
