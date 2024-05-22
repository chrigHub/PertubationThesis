python ./../../main/training/train.py \
  --dataspec "CREG" \
  --clf_type "RFR" \
  --est_type "EG" \
  --est_cv 5 \
  --est_scoring "neg_root_mean_squared_error" \
  --est_njobs 5 \
  --est_verbose 10 \
  \