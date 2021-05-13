def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Union[Tensor, Dict[str, Any]]:
    real_images, _ = batch
    batch_size = real_images.shape[0]
    # optimize Denerator and Discriminator networks
    return {
        'd_loss': d_loss,
        # other metrics to show on the determined dashboard
    }