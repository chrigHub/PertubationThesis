python ./../../main/training/train.py \
  --dataspec "CCLASS" \
  --clf_type "RF" \
  --est_type "EG" \
  --est_cv 5 \
  --est_scoring "accuracy" \
  --est_njobs 5 \
  --est_verbose 10 \
  \