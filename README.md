## 📊 Key Insights & Results
After engineering features for Size, Age, and Amenities, the analysis revealed:

1.  **Size Matters Most:** `TotalSF` (Total Square Footage) has the strongest correlation with price (**0.78**), validating that space is the primary value driver.
2.  **Age Penalty:** `HouseAge` shows a significant negative correlation (**-0.52**), indicating that for every year older a house is, its value depreciates reliably.
3.  **Bathrooms vs. Bedrooms:** `TotalBath` (**0.60**) is a better predictor of price than `BedroomAbvGr` (**0.16**). Modern buyers value luxury (bathrooms) over utility (just sleeping space).

**Model Performance (Random Forest):**
* **RMSE:** ~$30,000 (Average error per prediction)
* **R² Score:** 0.85 (The model explains 85% of the price variance)