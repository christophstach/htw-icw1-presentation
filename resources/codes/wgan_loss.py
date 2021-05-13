def discriminator_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    real_loss = -real_scores
    fake_loss = fake_scores

    loss = real_loss.mean() + fake_loss.mean()

    return loss

def generator_loss(fake_scores: Tensor) -> Tensor:
    fake_loss = -fake_scores
    loss = fake_loss.mean()

    return loss