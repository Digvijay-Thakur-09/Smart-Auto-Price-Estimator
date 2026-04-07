# Smart-Auto-Price-Estimator
Ford Price Predictor: A Scikit-Learn &amp; Streamlit project. Implements advanced data imputation, OneHotEncoding, and model serialization (Pickle) to deliver a blended ensemble price estimate.

2. Main Project Description (For the README)
Use this under your main title to explain the "Why" and "How" of your project.

Project Overview
This project provides a comprehensive solution for estimating the market value of used Ford vehicles. By bridging the gap between raw data and a user-friendly interface, it allows users to receive instant, data-driven price estimates based on historical market trends.

The Problem
Used car pricing is volatile and influenced by many non-linear factors (mileage, engine size, fuel type, and model popularity). A single model often carries bias; for instance, a Linear Regression might underperform on complex price drops in newer luxury models.

The Solution: The Ensemble Approach
Instead of relying on a single algorithm, this system utilizes an Ensemble Blending technique. It runs your inputs through four distinct models simultaneously:

Linear Regression: Establishes the baseline price trends.

K-Nearest Neighbors (KNN): Prices the car based on "neighbors" (similar cars in the dataset).

Decision Tree: Captures specific "if-then" market rules.

Random Forest: Reduces variance and provides the most robust individual prediction.

The final Estimated Price is a weighted average of these models, ensuring that the prediction is balanced and less prone to individual model errors.
