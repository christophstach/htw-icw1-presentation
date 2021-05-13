alpha = self.context.to_device(torch.rand(batch_size, 1, 1, 1))
interpolated_images = real_images + (1 - alpha) * fake_images
interpolated_images.requires_grad_(True)

scores = self.discriminator(interpolated_images)

ones = self.context.to_device(torch.ones_like(scores))
gradients = autograd.grad(outputs=scores, inputs=interpolated_images, grad_outputs=ones, create_graph=True)[0]
penalties = (gradients.norm(2, dim=1) - 1.0) ** 2
gradient_penalty = 10.0 * penalties.mean()