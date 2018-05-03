export BUCKET_NAME=livinglydev
export JOB_NAME="DFP_NN_test$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/ML/$JOB_NAME
export REGION=us-central1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/ML/$JOB_NAME \
  --runtime-version 1.0 \
  --module-name trainer.DFP_NN_cloud \
  --package-path /Users/jonas/Code/data/trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --train-file gs://livinglydev/ML/DFP_NN_000000000000.csv.gz
