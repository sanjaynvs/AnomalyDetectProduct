import pandas as pd
from sklearn.metrics import f1_score
import mlflow
from mlflow.models import infer_signature
from prodProcess import PredTask
from deeplog import Trainer, Predicter, Model, options, train, predict
from logdeep.models.lstm import Deeplog

# NOTE: review the links mentioned above for guidance on connecting to a managed tracking server, such as the Databricks Managed MLflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

# params = {"window_size": 10, "min_len":3,"vocab_size":23}

Model = Deeplog(input_size=options['input_size'],
                hidden_size=options['hidden_size'],
                num_layers=options['num_layers'],
                vocab_size=options["vocab_size"],
                embedding_dim=options["embedding_dim"])


trainer = Trainer(Model, options)
trainer.start_train()

pdt = Predicter(Model, options)
TH, TP, TN, FP, FN, P, R, F1 = pdt.predict_unsupervised()

with mlflow.start_run():
  # Log the hyperparameters
  mlflow.log_params(options)

  # Log the loss metric
  mlflow.log_metric("Best Threshold", TH)
  mlflow.log_metric("True Positive", TP)
  mlflow.log_metric("True Negative", TN)
  mlflow.log_metric("False Postive", FP)
  mlflow.log_metric("False Negative", FN)
  mlflow.log_metric("Precision", P)
  mlflow.log_metric("Recall", R)
  mlflow.log_metric("F1_Score", F1)

  # Set a tag that we can use to remind ourselves what this run was for
  mlflow.set_tag("Training Info", "Deeplog Model")

  # Infer the model signature
  signature = infer_signature( pdt.predict_unsupervised())

  # Log the model
  model_info = mlflow.sklearn.log_model(
      sk_model=pdt,
      name="Deeplog model",
      signature=signature,
    #   input_example=pred_sequences,
      registered_model_name="tracking-quickstart",
  )