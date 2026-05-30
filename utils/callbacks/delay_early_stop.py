from lightning.pytorch.callbacks import Callback


class DelayedEarlyStopping(Callback):
    def __init__(self, monitor="val_loss", patience=3, start_epoch=5):
        super(DelayedEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_score = None
        self.wait = 0

    def on_validation_batch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.start_epoch:
            return  # 前几个 epoch 不触发早停

        current_score = trainer.callback_metrics[self.monitor]
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True
        else:
            self.best_score = current_score
            self.wait = 0