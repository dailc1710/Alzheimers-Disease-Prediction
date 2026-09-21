# Data handling

`locked_test.csv` is a fixed internal evaluation cohort. It is created once by
the training pipeline, hashed into the artifact metadata, and excluded from
training and retraining. Do not replace it when comparing model versions.

The raw source CSV is kept outside the nested source package. Do not commit
uploaded patient data, temporary exports, or arbitrary serialized models.
