--------Skincare Hydration Prediction-------------

This project builds a machine learning model that predicts skin hydration score (0–100) based on common skincare ingredients.
It includes:
- A regression neural network trained in PyTorch
- Export to ONNX for browser-based inference
- A full NIO (Neural Input Optimization) module that "reverse engineers" ingredient values to reach a desired hydration level
- A clean interactive web UI built with HTML+ CSS + JavaScript + ONNX Runtime Web

-----Project Overview-------
Goal:
Use real skincare product ingredient data to:

- Train a regression model predicting hydration score

- Perform NIO optimization

- Freeze the model

- Optimize the inputs ingredients using gradient descent

- Find ingredient combinations that achieve a target hydration level

-----Dataset-------

Source:
Kaggle — Skincare Products & Ingredients
https://www.kaggle.com/datasets/eward96/skincare-products-and-their-ingredients?resource=download

Dataset includes:

- product name

- ingredient text

- product type

- price

- ingredient list

The project extracts binary chemical-ingredient features from raw text:

- hyaluronic_acid

- niacinamide

- retinol

- vitamin_c

- fragrance_free

- price_norm
Beginner-Friendly Explanation of Each Ingredient

Below we explain what each ingredient does, what typical safe ranges are, and how your example values fit into those ranges.
Everything is written in plain, simple language—no chemistry background needed.

------1. Hyaluronic Acid (%)----------
What it is:Hyaluronic Acid (often called “HA”) is a hydration magnet.
It pulls water into the skin and keeps it there, like a sponge.

Typical real-world range:
0.1% – 2%
Most drugstore moisturizers use around 0.1% – 0.5%
Serums specifically made for hydration may use 1–2%

How the model uses it:

Higher HA % → higher predicted hydration score.
But returns start to level off after ~1%.

-----2. Niacinamide (%)--------
What it is: Niacinamide is a form of Vitamin B3.
It helps:
Strengthen the skin barrier
Reduce redness
Improve hydration

Typical range:
2% (gentle formulas)
4–5% (common in moisturizers)
10% (serums, stronger products)

How the model uses it:
More niacinamide increases hydration effects up to a point.

-------3. Retinol (%)-------
What it is: Retinol is a Vitamin A derivative.
It improves:
Skin renewal
Collagen production
Texture over time
Typical range:
0.1% (beginner / mild)
0.3% (intermediate)
0.5–1% (advanced, strong)

How the model uses it:
Small amounts improve hydration long-term
Too high = hydration decreases (drying effect)

4. Vitamin C (%) 
What it is: Vitamin C brightens the skin, boosts collagen, and supports protection.

Typical range:
2–5% (mild, stable formulas)
10–20% (strong brightening serums)

How the model uses it:
Vitamin C interacts with Hyaluronic Acid to increase hydration in morning routines.

--------5. Fragrance Free?--------
What this means: The product contains added fragrance.

Why this matters:
Fragrance is one of the most irritating additives in skincare.
Even if it smells nice, it can:

Reduce hydration
Cause redness
Damage skin barrier (for sensitive people)

How the model uses it:

Fragrance = lower predicted hydration score
(Because irritation works against moisture retention.)

------6. Price per oz ($)------------
What “price per oz” means: This is the cost for one ounce of product.

It makes different products comparable.
For example:
Product A: $10 for 1 oz = $10/oz

Product B: $9 for 0.5 oz = $18/oz

So "price per ounce" gives a fair comparison of value.
Typical real-world ranges:

- $5–10/oz = drugstore
- $10–20/oz = mid-range
- $20–60/oz = prestige brands
- $60+/oz → luxury

How the model uses it:

Price has a small positive effect:
- Expensive products tend to have more active ingredients
- But the effect is limited — “more expensive” does not equal “more hydration”
- The model treats price mainly as a weak signal
- Price per oz exists in the model because hydration correlates slightly with quality and ingredient density.

----Model Architecture-----

Three models were evaluated:

Linear Regression

MLP Small (64 → 32 → 1)

MLP Large (128 → 64 → 32 → 1)

The best model (highest validation R²) is chosen automatically and exported to ONNX. 

------Training Results-------

The model is evaluated on:

MSE Loss

R² Score

The highest-performing model was then saved as:

"skincare_regression.onnx"

------Neural Input Optimization (NIO)-------

This model supports reverse prediction:

“Given a target hydration score, what combination of skincare ingredients achieves it?”

NIO works by:
- Freezing model weights
- Initializing ingredient values
- Treating the input vector as a learnable tensor
- Using gradient descent to minimize
- (predicted_hydration – target_hydration)²
Clamping results to biologically realistic ranges
This allows generation of:

- Budget-friendly routines
- High-performance routines
- Fragrance-tolerant formulas
- Ingredient-limited routines
