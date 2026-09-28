# KoAlign Synthetic Data

This directory contains the synthetic data constructed to augment the human-collected KoAlign dataset.

## Dataset Construction

The synthetic dataset was constructed through the following pipeline:

1. **Situation Construction**
   Human-collected emotion data were summarized into situation descriptions using Claude Opus.

2. **RoT Generation**
   A **Rule-of-Thumb (RoT) Generator**, fine-tuned on **KoAlign (Human)**, was used to generate RoTs for the constructed situations.

3. **Human-Guided Filtering**
   The generated data were filtered through:

   * heuristic-based filtering, and
   * manual filtering by one researcher.

   After filtering, **1,828 synthetic instances** were retained.

4. **Format Conversion**
   The filtered instances were converted into the **breakdown** and **free-form** formats used in KoAlign.

   This process resulted in a total of **7,312 free-form instances**.

## Dataset Statistics

The final synthetic dataset contains **7,312 free-form instances** with the following class distribution:

|     Class | # Instances |
| --------: | ----------: |
|        -1 |       1,168 |
|         0 |         452 |
|         1 |       5,692 |
| **Total** |   **7,312** |

## Discussion

The class distribution of the synthetic dataset is:

```text
{-1: 1168, 0: 452, 1: 5692}
```

This distribution may differ substantially from that of the original human-collected KoAlign dataset.

In particular, the synthetic dataset is skewed toward **class 1**. Therefore, this distributional difference should be taken into account when using the synthetic data for fine-tuning or when comparing models trained on the human and synthetic datasets.

## TODO

* [ ] Add code for dataset construction.
* [ ] Add detailed documentation for the dataset construction pipeline.
* [ ] Document the heuristic filtering criteria.
* [ ] Document the breakdown and free-form conversion procedures.
