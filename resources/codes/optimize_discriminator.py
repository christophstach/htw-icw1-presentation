self.discriminator.zero_grad()
z = self.context.to_device(torch.randn(batch_size, self.latent_dimension, 1, 1))

with torch.no_grad(): fake_images = self.generator(z)

real_scores = self.discriminator(real_images)
fake_scores = self.discriminator(fake_images)

d_loss = self.loss.discriminator_loss(real_scores, fake_scores)
gp = self.gradient_penalty(real_images, fake_images)

self.context.backward(d_loss + gp)
self.context.step_optimizer(self.opt_d)