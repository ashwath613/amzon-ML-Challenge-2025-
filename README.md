
# Smart Pricing – Multi-Modal Deep Learning Model (Hackathon Project, V5, RTX-4080)

## Project Overview

This repository contains the **Smart Pricing** model developed for a hackathon challenge as part of the **Artificial Intelligence and Data Science** program.
The objective of the project is to predict accurate product prices by learning from **catalog text**, **product images**, and **tabular features** such as packaging, brand, and quantity.

Our approach integrates **language models**, **computer vision**, and **numeric learning** into a single **multi-modal deep neural network** optimized for GPU inference.

---

## Team Members

* **Ashwath N**
* **Mahendra Kumar T**
* **Nithish Kumar B**

Institution: *Dr. Mahalingam College of Engineering and Technology*
Department: *B.Tech Artificial Intelligence and Data Science*

---

## Motivation

Modern e-commerce platforms list thousands of products, and manual price setting is inefficient and inconsistent. Our goal was to build an intelligent model capable of predicting fair and consistent product prices based on available catalog data, improving both pricing efficiency and business intelligence.

---

## Model Architecture

### Algorithms and Components Used

1. **Text Encoder – DeBERTa-v3-base**

   * Used for understanding product titles and descriptions
   * Extracts semantic meaning and brand-related patterns from catalog content

2. **Image Encoder – ConvNeXt-Tiny**

   * Processes product images and learns visual cues like size, packaging, and appearance
   * Pretrained on ImageNet-1K for general feature extraction

3. **Tabular Features (Numeric Data)**

   * Extracted from text using regular expressions
   * Includes: pack count, weight/volume, unit conversions, brand presence, and length statistics

4. **Fusion Technique – Gated Fusion Network**

   * Combines text, image, and tabular representations
   * Uses a learned gate mechanism to control contribution from each modality dynamically

5. **Regression Head**

   * Fully connected layers predict the **logarithm of price per unit (PPU)**
   * Final price is computed as:

     ```
     price = exp(predicted_log_ppu) * total_units
     ```

6. **Brand Priors (Post-Processing)**

   * Incorporates prior knowledge of brand-wise median PPU values for more stable predictions

---

## Methods and Workflow

1. **Data Preprocessing**

   * Automatic detection of units (g, ml, count)
   * Text normalization and brand extraction
   * Feature engineering using regex and linguistic cues

2. **Training Pipeline**

   * Multi-fold cross-validation (5 folds)
   * Mixed precision training (bfloat16/float16)
   * Loss: Mean Absolute Error (MAE) on log(price per unit)
   * Optimizer: AdamW with cosine learning rate schedule

3. **Inference Process**

   * Uses ensemble averaging across all folds
   * Blends model predictions with brand priors
   * Generates a final CSV file with predicted prices

---

## Input and Output

### Input Format

| Column            | Description                          |
| ----------------- | ------------------------------------ |
| `sample_id`       | Unique identifier for each product   |
| `catalog_content` | Product title or textual description |
| `image_link`      | Local image path or URL              |

### Output Format

| Column      | Description                         |
| ----------- | ----------------------------------- |
| `sample_id` | Product ID                          |
| `price`     | Predicted price (in local currency) |

Example output:

| sample_id | price |
| --------- | ----- |
| 100179    | 15.65 |
| 245611    | 14.82 |
| 146263    | 18.37 |
| 95658     | 10.59 |
| 36806     | 39.69 |

---

## How to Run

```python
from smart_pricing_infer_v5_4080 import run_predict

run_predict(
    input_path="dataset/test.xlsx",
    artifacts_dir="artifacts_v5_4080",
    output_csv="test_out.csv",
    batch_size=64,
    max_len=224
)
```

Console Output:

```
Device: cuda | bf16: True
✅ Wrote predictions to: test_out.csv
```

---

## Experience and Challenges

Building this project was a significant learning experience. Our team faced multiple challenges while tuning the model for high accuracy:

* **Data inconsistency:** Many catalog descriptions were unstructured, with missing units and brand names.
* **Image quality issues:** Low-resolution or missing images made visual inference difficult.
* **Model tuning:** Balancing contributions between text, image, and tabular inputs required careful experimentation.
* **Compute limitations:** Training large transformer-based models demanded efficient GPU memory optimization.
* **Ensembling and scaling:** Achieving stable results across multiple folds required precise scaling and normalization.

Despite these challenges, through continuous experimentation and teamwork, we achieved a **significant improvement in prediction accuracy** and developed a **robust, generalizable inference pipeline**.

---

## Results

* Model Version: V5
* Hardware: NVIDIA RTX-4080
* Architecture: DeBERTa + ConvNeXt + Gated Fusion
* Evaluation Metric: SMAPE (Symmetric Mean Absolute Percentage Error)
* Result: Stable predictions with reduced error variance across folds

---

## Future Work

* Integration of transformer-based image models (e.g., CLIP or ViT)
* Dynamic pricing through reinforcement learning
* Real-time web API deployment for business integration
* Incorporation of user sentiment and demand features

---

## License

This project is open-source and available under the **MIT License**.
It may be used for research, learning, and non-commercial applications with proper credit to the authors.

---

## Acknowledgement

Developed by the **Team SmartVision**
Department of Artificial Intelligence and Data Science
**Dr. Mahalingam College of Engineering and Technology**

Team Members:

* Ashwath N
* Mahendra Kumar T
* Nithish Kumar B

---

