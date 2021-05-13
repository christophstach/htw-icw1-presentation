self.generator.zero_grad()
z = torch.randn(batch_size, self.latent_dimension, 1, 1)
z = self.context.to_device(z)

fake_images = self.generator(z)
fake_scores = self.discriminator(fake_images)
g_loss = self.loss.generator_loss(fake_scores)

self.context.backward(g_loss)
self.context.step_optimizer(self.opt_g)