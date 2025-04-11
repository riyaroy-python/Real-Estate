# Real Estate Analysis & Prediction Project

This project performs an analysis of a real estate dataset and builds a model to predict the price of a property. The project includes:
- Data loading and initial exploratory data analysis (EDA)
- Data preprocessing (handling missing values, encoding categorical variables, scaling numeric features)
- Training a regression model (e.g., Linear Regression) to predict property prices
- An interactive Streamlit app for data visualization and prediction
- Unit tests for key modules
- Logging and error handling for robust code execution

## Project Structure

- **data/**: Contains raw and processed data (e.g., `real_estate.csv`)
- **src/**: Python modules:
  - `data_loader.py` – Loads the dataset and performs initial analysis
  - `predict.py` – Preprocesses the data (missing values, encoding, scaling)
  - `model.py` – Builds and evaluates a regression model
  - `utils.py` – Contains helper functions (e.g., saving plots)
- **app/**: Streamlit app (`streamlit.py`) for interactive data exploration and prediction
- **tests/**: Unit tests (using pytest)
- **logs/**: Directory where log files are stored
- **requirements.txt**: Python dependencies
- **.gitignore**: Specifies files and directories to ignore in version control

## Future Enhancements
- Add visualizations of prediction trends

## How to Run
git clone https://github.com/riyaroy-python/Real-Estate.git
   cd Real_Estat_project
