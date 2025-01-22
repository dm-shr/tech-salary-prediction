import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader

from src.training.transformer.dataset import DualTextDataset


def fit_eval_legacy(
    model,
    train_data,
    criterion,
    optimizer,
    config,
    device,
    logger,
    to_eval=True,
    test_data=None,
):
    """Train and evaluate the model."""
    train_dataset = DualTextDataset(**train_data)
    test_dataset = DualTextDataset(**test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    history = {"train_loss": [], "test_loss": [], "train_r2": [], "test_r2": []}

    for epoch in range(config["num_epochs"]):
        model.train()
        train_losses, test_losses = [], []
        all_preds, all_labels = [], []

        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs1, inputs2, targets = batch
            input1, attention_mask1 = inputs1["input_ids"].to(device), inputs1["attention_mask"].to(
                device
            )
            input2, attention_mask2 = inputs2["input_ids"].to(device), inputs2["attention_mask"].to(
                device
            )
            targets = targets.to(device)

            outputs = model(input1, attention_mask1, input2, attention_mask2).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

        train_r2 = r2_score(all_labels, all_preds)
        history["train_loss"].append(sum(train_losses) / len(train_losses))
        history["train_r2"].append(train_r2)

        if not to_eval:
            continue

        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                inputs1, inputs2, targets = batch
                input1, attention_mask1 = inputs1["input_ids"].to(device), inputs1[
                    "attention_mask"
                ].to(device)
                input2, attention_mask2 = inputs2["input_ids"].to(device), inputs2[
                    "attention_mask"
                ].to(device)
                targets = targets.to(device)

                outputs = model(input1, attention_mask1, input2, attention_mask2).squeeze()
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())

        test_r2 = r2_score(all_labels, all_preds)
        history["test_loss"].append(sum(test_losses) / len(test_losses))
        history["test_r2"].append(test_r2)
        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']}: Train R2={train_r2:.4f}, Test R2={test_r2:.4f}"
        )

    return model, history


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
