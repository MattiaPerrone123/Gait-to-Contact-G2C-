import matplotlib.pyplot as plt
import torch
import onnxruntime as ort

def train_one_epoch(model, train_dataloader, optimizer, criterion):
    """Perform one epoch of training"""
    model.train()
    epoch_loss=0

    for batch in train_dataloader:
        x_batch, y_batch=batch

        optimizer.zero_grad()
        output=model(x_batch)

        loss=criterion(output, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()

    return epoch_loss / len(train_dataloader)

def validate_one_epoch(model, test_dataloader, criterion):
    """Perform one epoch of validation"""
    model.eval()
    val_loss=0
    outputs=[]
    ground_truths=[]

    with torch.no_grad():
        for val_batch in test_dataloader:
            val_x_batch, val_y_batch=val_batch
            val_output=model(val_x_batch)

            loss=criterion(val_output, val_y_batch)
            val_loss+=loss.item()

            outputs.append(val_output.cpu().numpy())
            ground_truths.append(val_y_batch.cpu().numpy())

    return val_loss / len(test_dataloader), outputs, ground_truths

def save_model_onnx(model, input_sample, filepath='best_model.onnx'):
    """Save the model in ONNX format with dynamic batch size"""
    torch.onnx.export(
        model=model,
        args=input_sample,
        f=filepath,
        export_params=True,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )


def load_model_onnx(filepath):
    """Load the model using ONNX Runtime"""
    session=ort.InferenceSession(filepath)
    return session

def infer_onnx_model(session, input_tensor):
    """Perform inference with ONNX Runtime"""
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name
    result=session.run([output_name], {input_name: input_tensor.numpy()})
    return result[0]


def get_input_sample(dataloader, batch_size=1):
    """Retrieve an input sample from the dataloader"""
    input_sample, _ = next(iter(dataloader))
    return input_sample[:batch_size]


def plot_sample_images(outputs, ground_truths, epoch):
    """Plot output and ground truth images for visualization"""
    plt.figure(figsize=(10, 5))

    output_image=outputs[0][0].squeeze()
    ground_truth_image=ground_truths[0][0].squeeze()

    plt.subplot(1, 2, 1)
    plt.imshow(output_image, cmap='viridis')
    plt.title(f"Model Output (Epoch {epoch})")

    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth_image, cmap='viridis')
    plt.title(f"Ground Truth (Epoch {epoch})")

    plt.show()

def train_and_validate(model, train_dataloader, test_dataloader, num_epochs, optimizer, criterion, checkpoint_path, input_sample, plot_every=10):
    """Train and validate a model"""
    best_val_loss=float('inf')
    train_losses=[]
    val_losses=[]

    for epoch in range(1, num_epochs + 1):
        train_loss=train_one_epoch(model, train_dataloader, optimizer, criterion)
        train_losses.append(train_loss)

        val_loss, val_outputs, val_ground_truths=validate_one_epoch(model, test_dataloader, criterion)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            save_model_onnx(model, input_sample, filepath=checkpoint_path)

        print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {train_loss:.8f} Validation Loss: {val_loss:.8f}")

        if epoch % plot_every == 0:
            plot_sample_images(val_outputs, val_ground_truths, epoch)

    return train_losses, val_losses