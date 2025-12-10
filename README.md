Predicting telecom customer churn using PySpark and Machine Learning (Decision Tree, Random Forest, AdaBoost)

# Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-4.0.1-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📋 Project Overview

A telecommunications company is concerned about customers discontinuing their service (churn). This project performs **exploratory data analysis** and builds **machine learning models** to predict which customers are likely to churn, enabling proactive retention strategies.

## 🎯 Objectives

- Analyze customer behavior patterns that indicate potential churn
- Build and compare multiple ML classification models
- Identify key features that predict customer churn
- Provide actionable insights for customer retention

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **Records** | 3,333 observations |
| **Features** | 21 variables |
| **Target** | Churn (0 = Churner, 1 = Non-Churner) |

### Key Features

| Feature | Description |
|---------|-------------|
| `AccountLength` | How long account has been active |
| `DayMins` | Time spent on day calls |
| `EveMins` | Time spent on evening calls |
| `NightMins` | Time spent on night calls |
| `IntlMins` | Time spent on international calls |
| `CustServCalls` | Number of customer service calls |
| `IntlPlan` | International plan subscription |
| `VMailPlan` | Voicemail plan subscription |
| `DayCharge` | Charges for day calls |
| `EveCharge` | Charges for evening calls |
| `NightCharge` | Charges for night calls |
| `IntlCharge` | Charges for international calls |

## 🛠️ Technologies Used

- **PySpark 4.0.1** - Big data processing
- **Python 3.x** - Programming language
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models

## 📈 Exploratory Data Analysis

### Visualizations Created

1. **Histogram** - Day Minutes distribution by Churn status
2. **Count Plots** - Voicemail Plan vs Churn
3. **Count Plots** - International Plan vs Churn
4. **Bar Chart** - Area-wise Churner analysis
5. **Heatmap** - Correlation matrix

## 🤖 Machine Learning Models

Three classification models were trained and evaluated:

### Model Performance Comparison

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Decision Tree | 78.57% | 40% | 40% |
| **Random Forest** | **85.71%** | **100%** | 20% |
| AdaBoost | 82.14% | 50% | 40% |

## 🔍 Key Findings

1. **Random Forest** achieved the highest accuracy (85.71%) with perfect precision (100%)
2. **AdaBoost** provides balanced performance with better recall for identifying churners
3. Key predictors of churn include:
   - Day Minutes (`DayMins`)
   - Customer Service Calls (`CustServCalls`)
   - Day Charges (`DayCharge`)

## 💡 Recommendations

- Focus retention efforts on customers with high day minutes
- Monitor customers with multiple customer service calls
- AdaBoost or Decision Tree may be preferred for proactive churn identification due to higher recall

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/23396784/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

2. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn pyspark
```

3. Run the analysis:
```bash
python churn_analysis.py
```

Or use PySpark shell:
```bash
pyspark
```

## 📁 Project Structure

```
telecom-customer-churn-prediction/
│
├── README.md                 # Project documentation
├── Churn.csv                 # Dataset
├── churn_analysis.py         # Main analysis script
├── images/                   # Visualization outputs
│   ├── histogram_daymins.png
│   ├── countplot_vmailplan.png
│   ├── countplot_intlplan.png
│   ├── areawise_churn.png
│   └── correlation_matrix.png
└── results/                  # Model results
    └── model_comparison.csv
```

## 👨‍⚕️ Author

**Victor Prefa**

Medical Doctor | Data Scientist

- Integrating 17+ years of clinical experience with data science
- Specializing in healthcare analytics and predictive modeling
- MSc Data Science & Business Analytics

[![GitHub](https://img.shields.io/badge/GitHub-23396784-black.svg)](https://github.com/23396784)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/your-profile)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this project useful, please consider giving it a star!
