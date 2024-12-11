import matplotlib.pyplot as plt
import numpy as np
from .train_validate import infer_onnx_model


def plot_training_validation_loss(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()



def evaluate_model_onnx(session, dataloader, threshold=5e-4):
    """Evaluate the ONNX model on a DataLoader and collect predictions and ground truth"""
    predictions=[]
    ground_truth=[]
    
    for batch in dataloader:
        x_batch, y_batch=batch
        onnx_output=infer_onnx_model(session=session, input_tensor=x_batch)
        onnx_output[onnx_output < threshold]=0
        y_batch[y_batch < threshold]=0
        predictions.append(onnx_output)
        ground_truth.append(y_batch.cpu().numpy())
    predictions=np.concatenate(predictions, axis=0)
    ground_truth=np.concatenate(ground_truth, axis=0)
   
    return predictions, ground_truth



def plot_predictions_vs_ground_truth(predictions, ground_truth, num_samples=5):
    """Plot predictions and corresponding ground truth side by side"""
    for i in range(min(num_samples, predictions.shape[0])):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(predictions[i, 0], cmap='viridis')
        plt.title('Predicted')
        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth[i, 0], cmap='viridis')
        plt.title('Ground Truth')
        plt.show()
