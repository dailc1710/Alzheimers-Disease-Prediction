# Model Information Evidence Audit

## Executive finding

The Model information tab must explain the decision chain, not only print the active artifact's metrics. The aligned V3 report describes a compact five-feature XGBoost screening model, calibrated with sigmoid/Platt scaling and evaluated with a locked test set. The tab exposes the evidence for that decision, the columns that the active artifact depends on, and the release metadata used to refresh the report.

The aligned report and active artifact currently agree on threshold 0.425, the final XGBoost hyperparameters (`learning_rate=0.03`, `max_depth=3`, `n_estimators=300`), the selected `scale_pos_weight` strategy, the feature set, and the stored evaluation metrics. The reference values in the UI remain a snapshot so a later retrain is visible as a new release instead of silently rewriting the historical comparison.

## Evidence from the V3 report

The supplied `../../deliverables/reports/I1_2510_E0_-_Nhom_1-_Alzheimer_project_report_V3_revised.docx` supports the following chain:

1. The source has 2,149 rows and 32 candidate features after excluding identifiers and the constant `DoctorInCharge`. The BP rule `SystolicBP <= DiastolicBP` removes 186 rows, leaving 1,963 rows for modeling.
2. Permutation importance and feature-count comparisons retain five official features: `MMSE`, `FunctionalAssessment`, `ADL`, `MemoryComplaints`, and `BehavioralProblems`.
3. The report compares DummyClassifier, Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting, and XGBoost with repeated stratified cross-validation. XGBoost is selected among the real candidates because it has the highest reported PR-AUC, retains recall comparable to Random Forest, and exposes `scale_pos_weight` for class imbalance.
4. The final model is calibrated with Platt scaling and the threshold is selected by F2 on validation. The report evaluates once on the locked test set and reports bootstrap confidence intervals.

The aligned feature-selection table is nuanced: five features score accuracy 0.9530 and PR-AUC 0.9394, while eight features have slightly higher PR-AUC (0.9402) and all 32 score 0.9427/0.9361. Therefore the official five-feature choice is a reproducibility, compactness and interpretability decision, not proof that excluded columns have no predictive relationship.

## Active artifact evidence

The finalized `artifacts/active_metadata.json` records:

- XGBoost benchmark PR-AUC CV 0.9394, ROC-AUC CV 0.9588, F1 0.9337, and recall 0.9347.
- Five selected features identical to the official V3 set.
- Threshold 0.425 and sigmoid calibration.
- 2,149 raw rows and 1,963 clean rows after BP filtering.
- Hyperparameters `learning_rate=0.03`, `max_depth=3`, and `n_estimators=300`, with `scale_pos_weight=1.8288`.

The active artifact's imbalance comparison is not a universal win for `scale_pos_weight`: SMOTENC is slightly higher on PR-AUC, the Baseline has the lowest Brier score, and `scale_pos_weight` has the highest recall and F2. The UI therefore describes the choice as recall-oriented for screening and displays the comparison as a two-panel chart: screening metrics on the left and Brier calibration error on the right.

## Which columns influence the model

The primary model-level evidence is validation permutation importance. In the pipeline, the model is fit on the training split and each validation column is shuffled repeatedly; the stored `importance_mean` is the mean decrease in validation accuracy. A larger positive value means the fitted model relies more on that column for that validation score. This is a dependence measure, not a causal or clinical effect. Correlated columns can share, mask, or redistribute importance.

The tab also shows Cohen's d from the EDA artifact as a separate univariate signal. Cohen's d answers a different question: how separated the two Diagnosis groups are for one feature. It must not be read as a model coefficient, a risk ratio, or a statement that changing a feature causes disease.

The current active ranking places the five final features at the top of the stored validation ranking. The tab shows both the mean and standard deviation of permutation importance, then merges Cohen's d when available. `PatientID` and `DoctorInCharge` are explicitly listed as excluded: the former is an identifier and the latter is constant in the source data.

## UI changes required by this audit

- Show a clearly labelled **Model comparison** chart near the top of the tab, with the selected model highlighted; keep exact values in a collapsed audit expander and explain the primary metric.
- Show a V3-report-versus-active-artifact evidence table for benchmark values, threshold, and important hyperparameters.
- Show the feature ranking as a chart plus a table containing mean importance, standard deviation, and Cohen's d when available.
- Explain why five features were retained versus all 32, including the accuracy/PR-AUC tradeoff.
- Show the imbalance-method comparison and state which metric justifies the recorded choice.
- State that permutation importance is model dependence, not causality, and that the model is a screening demonstration rather than a diagnostic device.
- Surface artifact/retraining mismatches instead of presenting the active artifact as an exact copy of the report.

## Additional research findings for the Model information tab

### Model card and intended use

The tab should make the model's purpose and output explicit before showing performance numbers. The output is a calibrated probability-like screening score and a thresholded screen label for the dataset target `Diagnosis`; it is not a diagnosis, treatment recommendation, triage decision, or evidence of clinical utility. This distinction follows the transparency principle that users need to understand intended use, inputs, outputs, workflow placement, limitations, and known gaps before acting on a machine-learning result.[6]

The implementation now shows the artifact version, random state, train/validation/test counts, grid-search candidate count, selected features, threshold, calibration method, and current data-cleaning summary. These fields are reproducibility and audit context; they do not replace external validation.

### Operating point and confusion matrix

Accuracy and F1 do not show the error trade-off at the chosen threshold. The V3 report gives a locked-test confusion matrix of TN=242, FP=12, FN=13, TP=126 for `n=393`. The UI now renders this as a heatmap and derives sensitivity/recall, specificity, PPV/precision, and NPV from the counts. These metrics are threshold-dependent; ROC-AUC and PR-AUC are not.

The training pipeline persists the test confusion matrix and the active
artifact uses threshold 0.425. The aligned report is refreshed from the same
metadata, so the UI and DOCX show the same operating point.

### Why this model and which columns

The model decision is now presented as an evidence chain. The experiment starts with two baselines (DummyClassifier and Logistic Regression) and five candidate models, then uses repeated stratified cross-validation. Because the cleaned data has a positive rate of about 35.4%, PR-AUC is used as the primary ranking metric and recall is shown as the screening-oriented trade-off. In the active benchmark, XGBoost has the highest stored PR-AUC CV (0.9394), while Random Forest is slightly better on ROC-AUC; therefore XGBoost is selected for the explicit PR-AUC/recall policy, not because it wins every metric.

The final input columns are `MMSE`, `FunctionalAssessment`, `ADL`, `MemoryComplaints`, and `BehavioralProblems`. The tab now gives each column a plain-language source-field role and shows its validation permutation importance, with Cohen's d and diagnosis-group means available in an audit expander. These are separate signals: permutation importance describes dependence of the fitted model on validation accuracy, while Cohen's d and group means describe univariate group separation. Neither is a causal explanation, and the app labels the model as structured/tabular XGBoost rather than a text-generating LLM.

### Calibration evidence

TRIPOD+AI treats discrimination, calibration and clinical utility as distinct parts of prediction-model evaluation.[7] The aligned active artifact records validation Brier improving from 0.0696 to 0.0642 and quantile-10-bin ECE from 0.0735 to 0.0480 after sigmoid/Platt calibration. The tab and DOCX present those same saved values, then separately show the locked-test Brier score.

The scikit-learn calibration display documentation describes a calibration curve/reliability diagram as the relationship between mean predicted probability and the observed fraction of positives in bins.[8] The active metadata also stores the validation binned calibration points for audit.

### Fairness, transportability and lifecycle gaps

TRIPOD+AI specifically calls for fairness considerations and performance evaluation in key subgroups, while also emphasizing representative evaluation data and open-science details such as code, data and model availability.[7] The FDA transparency principles likewise recommend communicating intended population, inputs/outputs, model and dataset characteristics, confidence intervals, known biases or failure modes, gaps in data representation, and plans for ongoing performance monitoring.[6]

The current artifact stores subgroup metrics, internal monitoring baselines,
and calibration summaries, but external validation and decision-curve analysis
have not been performed. The tab exposes those remaining governance gaps rather
than presenting overall metrics as evidence of clinical usefulness. These are
follow-up evaluation requirements, not values that can be safely fabricated
from the current metadata.

### Implementation status

- Added model-card summary: intended use, output semantics, out-of-scope uses and reproducibility context.
- Added locked-test operating-point visualization and threshold-dependent metrics.
- Added active-artifact calibration before/after visualization with saved Brier and quantile-10-bin ECE.
- Added artifact persistence for `test_confusion_matrix` in the active release.
- Added a governance expander documenting external-validation, decision-curve and clinical-utility gaps while showing available subgroup and monitoring evidence.
- Moved the model-comparison chart to the top of Model information so the benchmark is discoverable without scrolling through audit sections.
- Added an explicit model-selection narrative and a dedicated chart for the five final input columns, with exact feature evidence available on demand.
- Extended the source list with TRIPOD+AI, scikit-learn calibration/confusion-matrix documentation and FDA transparency guidance.

## Sources

1. Supplied project report: `../../deliverables/reports/I1_2510_E0_-_Nhom_1-_Alzheimer_project_report_V3_revised.docx`, sections 6, 8, 10, 11, 12 and 14; tables covering feature selection, imbalance comparison, model benchmark, calibration, test metrics and validator behavior.
2. XGBoost documentation, “XGBoost Parameters,” especially `scale_pos_weight`: https://xgboost.readthedocs.io/en/latest/parameter.html
3. scikit-learn documentation, “CalibratedClassifierCV”: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV
4. scikit-learn documentation, “Permutation Importance vs Random Forest Feature Importance”: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
5. scikit-learn documentation, “Precision-Recall”: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
6. U.S. FDA, Health Canada and MHRA, “Transparency for Machine Learning-Enabled Medical Devices: Guiding Principles”: https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles
7. Collins GS et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” BMJ 2024;385:e078378: https://www.bmj.com/content/385/bmj.q902
8. scikit-learn documentation, “CalibrationDisplay”: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html
9. scikit-learn documentation, “confusion_matrix”: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
