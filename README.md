# The Pembrook Papers: Muggle Analytics Applied to Magical Problems
### A Private Research Initiative — *This Repository Does Not Exist*

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Analysis-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Classified-darkred)

---

## About the Analyst

**Reginald Pembrook**, Senior Archivist  
Department of Magical Records, Ministry of Magic, Whitehall  
Ravenclaw, Class of 1991  
Wand: Sycamore & Unicorn Hair, 14 inches, Very Flexible

*"The sycamore makes a questing wand, eager for new experience and losing brilliance if engaged in mundane activities."* — Garrick Ollivander

On 14 March 2019, a Muggle-born intern left a laptop in the Ministry break room. Pembrook's wand had nearly combusted twice that month from filing Floo Network usage reports. Fourteen hours later, he had installed Python and written his first for loop. His wand glowed warmly for the first time in eleven years.

This notebook is the result of five years of unauthorized cross-disciplinary research, conducted behind a Glamour Charm, presented here as evidence that curiosity is not, in fact, a disciplinary offence.

---

## What This Project Actually Demonstrates

Behind the wizarding world framing, this is a portfolio of five complete machine learning and statistical analyses:

| Analysis | Magical Question | Real Data Science |
|---|---|---|
| 1. Lost Charm Recovery | Can linguistic structure predict spell effectiveness? | Random Forest Regression, feature importance |
| 2. Herbology Clustering | Do magical plants group by functional property? | K-Means Clustering, StandardScaler, elbow method |
| 3. Dark Arts Detection | Can we flag dark wizard activity before it escalates? | Z-score anomaly detection, time series analysis |
| 4. Potions Optimization | What actually determines brewing success? | Random Forest Classification, class report |
| 5. Alchemy Modeling | Is the Philosopher's Stone really necessary? | Logistic Regression, interaction effects |

---

## Key Findings

**Charm Recovery:** Vowel count and terminal phoneme ('a' ending) predict effectiveness better than syllable count alone. 28 of 78 lost charms are predicted at ≥70 effectiveness — worth reconstructing.

**Herbology:** Four natural plant clusters emerge: Benign & Common, Lunar-Sensitive, High Potency/Dangerous, and Rare & Complex. High magical potency correlates strongly with toxicity (r≈0.71).

**Dark Arts:** A statistical early-warning system would have flagged the 1995-97 activity surge approximately 6 months before the Ministry's owl-post network. The 2024 trend warrants monitoring.

**Potions:** Brewer skill level and ingredient quality account for ~60% of predictive importance. Temperature has a meaningful window (85-95°C optimal). Clockwise stirring count matters less than commonly taught.

**Alchemy:** A Grand Master alchemist without a Philosopher's Stone achieves higher success rates than an Apprentice with a genuine stone. The Stone is a multiplier, not a prerequisite.

---

## Why This Project Exists

Every portfolio has a data cleaning project. Every portfolio has a churn analysis. This portfolio has those — and it also has a Ravenclaw archivist who taught himself machine learning because his wand was about to combust from boredom.

The analysis is real. The methods are real. The datasets were generated with historically-grounded synthetic data. The wizard is fictional.

The skill demonstrated — applying the full ML workflow (data generation, EDA, feature engineering, model selection, evaluation, communication) to an unfamiliar domain and explaining it clearly — is not.

---

## Technical Stack

```
Python 3.11
pandas, NumPy — data manipulation
scikit-learn — RandomForest, KMeans, LogisticRegression, StandardScaler
matplotlib, seaborn — visualization (dark theme throughout)
```

---

## Files

```
pembrook/
├── pembrook_magical_analytics.ipynb  # Main analysis notebook (18 cells)
├── generate_data.py                  # Synthetic data generator
├── README.md
└── data/
    ├── charm_manuscripts.csv         # 250 ancient spell records
    ├── herbology_properties.csv      # 30 magical plant specimens
    ├── dark_arts_incidents.csv       # 2,777 Ministry incident reports (1990-2024)
    ├── potions_brewing_log.csv       # 500 brewing attempts
    └── alchemy_transmutation_log.csv # 300 transmutation records
```

---

## To Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python3 generate_data.py  # Creates the data/ folder
jupyter notebook pembrook_magical_analytics.ipynb
```

*Note: If your supervisor asks what you're working on, tell them it's a new cataloguing system for the Restricted Section. It's not entirely untrue.*

---

## Author

**Alejandro (Alex) Molina**  
GitHub: [@chooseyourtacoday](https://github.com/chooseyourtacoday)  
Location: Dallas–Fort Worth, TX  
Certifications: Codecademy BI Data Analyst | Databricks Analyst Associate (In Progress)

*Pembrook is fictional. The Python is not.*
