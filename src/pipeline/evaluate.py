from sklearn.metrics import fbeta_score, precision_score, recall_score

from pipeline.model import inference_model


def compute_metrics(y_true, y_pred):

    f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)

    return f1, precision, recall

def evaluate(file, model_pipe, X, y, split):

    print("[INFO] Running inference")
    y_pred = inference_model(model_pipe, X)
    
    print("[INFO] Evaluating model")
    pre, rec, f1 = compute_metrics(y_pred, y)
    
    print(f"[INFO] Evalating {split} data")
    print("Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}".format(
        pre, rec, f1))

    print(f"Evaluation on {split} data", file=file)
    print(f"Precision = {pre:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}", file=file)
    
