python ./../../main/training/train.py \
  --dataspec "B" \
  --clf_type "KNN" \
  --est_type "SH" \
  --est_cv 5 \
  --est_scoring "accuracy" \
  --est_njobs 5 \
  --est_verbose 10 \
  \