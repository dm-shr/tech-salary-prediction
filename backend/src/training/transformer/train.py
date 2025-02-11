import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from src.training.transformer.dataset import DualTextDataset
from torch.utils.data import DataLoader


def fit_eval(
    model, train_data, criterion, optimizer, config, device, logger, to_eval=True, test_data=None
):
    """Train and evaluate the model with expanded metrics tracking."""
    train_config = config["models"]["transformer"]
    train_dataset = DualTextDataset(**train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
    )

    if to_eval:
        test_dataset = DualTextDataset(**test_data)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
        )

    # Initialize history with all metrics
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_rmse": [],
        "test_rmse": [],
        "train_r2": [],
        "test_r2": [],
        "train_mae": [],
        "test_mae": [],
        "y_pred": [],
        "y_test": [],  # These will be lists of lists (one per epoch)
    }

    num_epochs = (
        train_config["num_epochs"] if not config["is_test"] else train_config["num_epochs_test"]
    )

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_predictions = []
        train_actuals = []

        # Training loop
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs1, inputs2, targets = batch
            input1, attention_mask1 = inputs1["input_ids"].to(device), inputs1["attention_mask"].to(
                device
            )
            input2, attention_mask2 = inputs2["input_ids"].to(device), inputs2["attention_mask"].to(
                device
            )
            # targets = targets.to(device)
            targets = targets.to(
                device
            ).squeeze()  # Squeeze the targets to match the shape of outputs

            outputs = model(input1, attention_mask1, input2, attention_mask2).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_actuals.extend(targets.cpu().numpy())

        # Calculate training metrics
        train_loss = sum(train_losses) / len(train_losses)
        train_rmse = root_mean_squared_error(train_actuals, train_predictions)
        train_r2 = r2_score(train_actuals, train_predictions)
        train_mae = mean_absolute_error(train_actuals, train_predictions)

        # Store training metrics
        history["train_loss"].append(train_loss)
        history["train_rmse"].append(train_rmse)
        history["train_r2"].append(train_r2)
        history["train_mae"].append(train_mae)

        if to_eval:
            model.eval()
            test_losses = []
            test_predictions = []
            test_actuals = []

            with torch.no_grad():
                for batch in test_dataloader:
                    inputs1, inputs2, targets = batch
                    input1, attention_mask1 = inputs1["input_ids"].to(device), inputs1[
                        "attention_mask"
                    ].to(device)
                    input2, attention_mask2 = inputs2["input_ids"].to(device), inputs2[
                        "attention_mask"
                    ].to(device)
                    # targets = targets.to(device)
                    targets = targets.to(
                        device
                    ).squeeze()  # Squeeze the targets to match the shape of outputs

                    outputs = model(input1, attention_mask1, input2, attention_mask2).squeeze()
                    loss = criterion(outputs, targets)

                    test_losses.append(loss.item())
                    test_predictions.extend(outputs.cpu().numpy())
                    test_actuals.extend(targets.cpu().numpy())

            # Calculate test metrics
            test_loss = sum(test_losses) / len(test_losses)
            test_rmse = root_mean_squared_error(test_actuals, test_predictions)
            test_r2 = r2_score(test_actuals, test_predictions)
            test_mae = mean_absolute_error(test_actuals, test_predictions)

            # Store test metrics and predictions for this epoch
            history["test_loss"].append(test_loss)
            history["test_rmse"].append(test_rmse)
            history["test_r2"].append(test_r2)
            history["test_mae"].append(test_mae)
            history["y_pred"].append(test_predictions)
            history["y_test"].append(test_actuals)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
                f"Train R2={train_r2:.4f}, Test R2={test_r2:.4f}, "
                f"Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}, "
                f"Train MAE={train_mae:.4f}, Test MAE={test_mae:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, "
                f"Train R2={train_r2:.4f}, "
                f"Train RMSE={train_rmse:.4f}, "
                f"Train MAE={train_mae:.4f}"
            )

    return model, history


def predict(model, inputs1, inputs2):
    """
    Predict target value for a single pair of preprocessed inputs.

    Args:
        model: Trained transformer model
        inputs1: First preprocessed input dictionary with 'input_ids' and 'attention_mask'
        inputs2: Second preprocessed input dictionary with 'input_ids' and 'attention_mask'

    Returns:
        float: Predicted value
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Move inputs to device
    input1 = inputs1["input_ids"].to(device)
    attention_mask1 = inputs1["attention_mask"].to(device)
    input2 = inputs2["input_ids"].to(device)
    attention_mask2 = inputs2["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input1, attention_mask1, input2, attention_mask2).squeeze()

    return output.item()
