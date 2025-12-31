# RailRadar Imputation Validation Summary

## Metrics on labeled set (100 examples)
   stat       mae      rmse  coverage_1std  coverage_1.96std  coverage_2std   z_mean    z_std
rr_mean 36.873688 58.187177           0.71              0.89            0.9 0.450945 2.435430
 rr_std 21.146864 47.971666           0.69              0.90            0.9 0.408351 3.781089

## Distribution comparison
   stat  label_mean  label_median  label_std  pred_lab_mean  pred_lab_std  pred_all_mean  pred_all_std  ks_stat    ks_pvalue
rr_mean   44.800603     26.040000  52.103300      40.215012     25.789171      31.521147     15.085002 0.322134 2.300153e-08
 rr_std   28.326488     18.445591  46.244352      26.574051     12.373974      20.957880      7.909509 0.369019 6.730854e-11

High-uncertainty imputations: 581 rows written to `reports/high_uncertainty_imputations.csv`


## Calibration and Flagging
Applied normalized-residual calibration using q95 scales: mean_scale=4.515, std_scale=4.208.
Flagged 581 trains as high-uncertainty; CSV: reports/high_uncertainty_imputations_calibrated.csv
