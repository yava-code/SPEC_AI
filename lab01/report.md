# lab 1: environment and first system measurements

## 1. goal

the goal of this laboratory work is to configure an isolated and reproducible python environment, train baseline logistic regression and random forest classification models on the breast cancer wisconsin dataset, empirically measure training duration, single-sample inference latency, system memory allocation, and serialization footprints, and evaluate deployment feasibility across cloud, edge, mobile, and tinyml resource constraints.

## 2. method

reproducibility was maintained across all steps by fixing pseudo-random seeds to 42. the breast cancer wisconsin diagnostic dataset was partitioned into stratified training and testing splits using a seventy-thirty ratio. baseline models including logistic regression with a limit of one thousand iterations and a random forest classifier configured with one hundred estimators were fitted to the training split. timing benchmarks followed a strict protocol consisting of an initial warm-up execution followed by repeated iterations to record median values. training duration reflects the median of five consecutive runs. single-sample inference latency reflects the median across one hundred executions. model artifacts were serialized to disk using joblib, and peak process resident memory was monitored via psutil routines.

## 3. results

| model | accuracy | train time (s) | peak train ram (mb) | inference latency (ms) | model size (bytes) | model size (kb) |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.9415 | 1.44444 | 0.0039 | 0.8901 | 1103 | 1.08 |
| RandomForest | 0.9357 | 0.33573 | 0.0039 | 14.0634 | 290921 | 284.10 |

## 4. three conclusions

logistic regression easily satisfies all tinyml operational budgets because its single-sample inference latency of zero point eighty-nine milliseconds remains well below the ten millisecond threshold and its serialized size of one point zero eight kilobytes comfortably fits within the one hundred kilobyte storage cap.

random forest violates the strict tinyml resource envelope by exceeding the storage threshold with a footprint of two hundred eighty-four kilobytes against the one hundred kilobyte limit and by posting an inference latency of fourteen milliseconds against the ten millisecond target, yet it easily clears all constraints required for deployment on edge and mobile targets.

system-level performance analysis demonstrates the practical trade-off between architectural complexity and resource efficiency where logistic regression delivers marginally better classification accuracy on this dataset while achieving roughly fifteen times faster single-sample inference and demanding over two hundred sixty times less persistent storage than the ensemble alternative.