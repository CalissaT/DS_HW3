# Analysis of Regional Video Game Sales Predictions

## Overall Comments
- Linear models were evaluated with R² metric from 10-fold cross validation (better performing than the tree models). Tree models were used to give an explicit percentage for importance of the feature. Linear model weights can be used to get a relative importance compared to each other because of min-max normalization on the features for that model.

## NA Sales Prediction

### Feature Importances:
- EU_Sales: 77.3% importance
- JP_Sales: 22.7% importance

### Linear Regression Weights:
- EU_Sales: 33.10
- JP_Sales: 3.84

### Cross-Validation Results:
- Overall model (EU + JP): R² = -20.52
- EU alone: R² = -27.05
- JP alone: R² = -123.60

### Key Findings:
EU sales is the dominant predictor for NA sales (explains 77% of the sales with a tree model). However, all models show negative R² scores, 
indicating they perform worse than predicting with the mean game sales value. Using both features together 
provides the best (least negative) result.

---

## EU Sales Prediction

### Feature Importances:
- NA_Sales: 84.0% importance
- JP_Sales: 16.0% importance

### Linear Regression Weights:
- NA_Sales: 18.40
- JP_Sales: 1.89

### Cross-Validation Results:
- Overall model (NA + JP): R² = -1.90
- NA alone: R² = -2.60
- JP alone: R² = -43.62

### Key Findings:
NA sales is by far the most important predictor for EU sales (explains 84% with a tree model). The negative R² scores are 
much smaller in magnitude than NA predictions, suggesting EU sales are somewhat more predictable. 
The combined model performs significantly better than using individual features.

---

## JP Sales Prediction

### Feature Importances:
- NA_Sales: 55.1% importance
- EU_Sales: 44.9% importance

### Linear Regression Weights:
- NA_Sales: 4.42
- EU_Sales: 3.90

### Cross-Validation Results:
- Overall model (NA + EU): R² = -1.39
- NA alone: R² = -1.68
- EU alone: R² = -2.41

### Key Findings:
JP sales has the most balanced feature importance between NA and EU sales (approximately 55%-45% split for a tree model). 
This model has the best cross-validation scores (least negative), suggesting JP sales are the most 
predictable of the three regions. Both NA and EU contribute meaningfully to the prediction.

---

## Overall Conclusions

1. **Negative R² scores** indicate all linear models perform poorly, this is due to the low amount of features used in the models. However, the intentions of this model was to see the most correlated regions not necessarily to predict most accurately (to do that we would include other features)
2. **Regional interdependence**: Each region's sales correlate with others, but the relationships 
   are complex (NA predicts EU way better than EU predicts NA (as seen in big negative R² and lower feature importance))
3. **Best predictions**: JP sales > EU sales > NA sales (in terms of model performance)
4. **Dominant predictors**: EU -> NA (77%), NA -> EU (84%), about balanced for JP (~55% for NA predicting JP and ~45 for EU predicting JP)
