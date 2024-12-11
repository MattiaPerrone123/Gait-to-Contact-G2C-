import pytorch_lightning
import torch
import optuna
from .model import UNetDecoder
from .model import TimeSeriesTransformerEncoder
from torch.nn import functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class LitTimeSeriesToImageModel(pytorch_lightning.LightningModule):
    def __init__(self, input_dim=128, hidden_dim=256, nheads=2, num_layers=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder=TimeSeriesTransformerEncoder(input_dim=self.hparams.input_dim,
                                                   hidden_dim=self.hparams.hidden_dim,
                                                   nheads=self.hparams.nheads,
                                                   num_layers=self.hparams.num_layers)

        self.decoder=UNetDecoder(in_channels=self.hparams.input_dim)

    def forward(self, src):
        src=self.encoder(src)
        src=src.view(src.size(0), self.hparams.input_dim, 10, 10)
        return self.decoder(src)

    def training_step(self, batch, batch_idx):
        x, y=batch
        y_hat=self(x)
        loss=F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y=batch
        y_hat=self(x)
        val_loss=F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def objective(trial, train_dataloader, test_dataloader):
    input_dim=trial.suggest_categorical('input_dim', [64, 128, 256])
    possible_nheads=[n for n in [1, 2, 4, 8] if input_dim % n == 0]
    nheads=trial.suggest_categorical('nheads', possible_nheads)
    hidden_dim=trial.suggest_categorical('hidden_dim', [128, 256, 512])
    num_layers=trial.suggest_int('num_layers', 1, 4)
    learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)

    model=LitTimeSeriesToImageModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nheads=nheads,
        num_layers=num_layers,
        learning_rate=learning_rate
    )
    logger=TensorBoardLogger(save_dir="tb_logs", name="my_model")

    early_stop_callback=EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")
    checkpoint_callback=ModelCheckpoint(dirpath="checkpoints", monitor="val_loss", mode="min", save_top_k=1)

    trainer=Trainer(logger=logger, callbacks=[early_stop_callback, checkpoint_callback], max_epochs=50)

    trainer.fit(model, train_dataloader, test_dataloader)

    val_loss=trainer.callback_metrics.get('val_loss', float('nan'))
    return val_loss


def run_optuna_study(train_dataloader, test_dataloader):
    study=optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataloader, test_dataloader), n_trials=100)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial=study.best_trial

    print("    Value: ", trial.value)
    print("    Parameters: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    return study
