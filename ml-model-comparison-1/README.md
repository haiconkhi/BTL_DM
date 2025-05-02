# Machine Learning Model Comparison

This project implements and compares the performance of four different machine learning models using two datasets: `Marketing_end.csv` and `marketing_campaign.csv`. The models included in this project are:

1. **Artificial Neural Network (ANN)**
2. **Gradient Boosting**
3. **Linear Regression**
4. **XGBoost**

## Project Structure

```
ml-model-comparison
├── data
│   ├── Marketing_end.csv
│   └── marketing_campaign.csv
├── src
│   ├── compare_models.py
│   └── models
│       ├── ann.py
│       ├── gradient_boosting.py
│       ├── linear_regression.py
│       └── xgboost.py
├── requirements.txt
└── README.md
```

## Requirements

To run this project, you need to install the following dependencies:

- pandas
- scikit-learn
- xgboost
- keras
- tensorflow

You can install the required packages using the following command:

```
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Ensure that the datasets `Marketing_end.csv` and `marketing_campaign.csv` are located in the `data` directory.

2. **Running the Comparison**: Execute the `compare_models.py` script to load the datasets, preprocess the data, train the models, and compare their performance.

   ```
   python src/compare_models.py
   ```

3. **Interpreting Results**: After running the script, the performance metrics of each model will be displayed in the console. You can analyze these metrics to determine which model performs best on the given datasets.

## Conclusion

This project serves as a foundation for comparing different machine learning models. You can extend it by adding more models, tuning hyperparameters, or using different datasets.