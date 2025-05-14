import torch.nn as nn
import torch


class GAN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        latent_dim,
        num_classes,
        label_emb_dim,
        device,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.generator = Generator(
            latent_dim + label_emb_dim,
            hidden_dims,
            output_dim=input_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).to(device)

        self.discriminator = Discriminator(
            input_dim + label_emb_dim,
            list(reversed(hidden_dims)),
            output_dim=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).to(device)

        self.label_embedding = LabelEmbedding(
            num_classes=num_classes, embedding_dim=label_emb_dim
        ).to(device)
        self.device = device
        self.num_classes = num_classes
        self.latent_dim = latent_dim

    def generate(self, y):
        z = torch.randn((y.shape[0], self.latent_dim), device=self.device)
        y_emb = self.label_embedding(y)
        z_cond = torch.cat([z, y_emb], dim=1)
        return self.generator(z_cond)

    def discriminate(self, x, y):
        y_emb = self.label_embedding(y)[:, :, None, None].expand(
            -1, -1, x.shape[2], x.shape[3]
        )
        x_cond = torch.cat([x, y_emb], dim=1)
        return self.discriminator(x_cond)

    def save(self, path):
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "label_embedding": self.label_embedding.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.label_embedding.load_state_dict(ckpt["label_embedding"])


class Generator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        kernel_size,
        stride,
        padding,
    ):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.proj = self.projection_block(input_dim)
        self.blocks = self.build_generator()

    def build_generator(self):
        blocks = nn.ModuleList()
        hidden_in = self.hidden_dims[0]
        for hidden_out in self.hidden_dims[1:-1]:
            blocks.append(self.generator_block(hidden_in, hidden_out))
            hidden_in = hidden_out
        blocks.append(self.generator_final(hidden_in, self.output_dim))
        return nn.Sequential(*blocks)

    def projection_block(self, input_dim):
        return nn.Sequential(
            nn.Linear(
                input_dim, self.hidden_dims[0] * self.kernel_size * self.kernel_size
            ),
            nn.ReLU(),
        )

    def generator_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, self.kernel_size, self.stride, self.padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def generator_final(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, self.kernel_size, self.stride, self.padding
            ),
            nn.Tanh(),
        )

    def forward(self, z_cond):
        x = self.proj(z_cond).view(
            z_cond.shape[0], -1, self.kernel_size, self.kernel_size
        )
        return self.blocks(x)


class Discriminator(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, kernel_size, stride, padding
    ):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.blocks = self.build_discriminator()

    def build_discriminator(self):
        blocks = nn.ModuleList()
        hidden_in = self.input_dim
        for hidden_out in self.hidden_dims:
            blocks.append(self.discriminator_block(hidden_in, hidden_out))
            hidden_in = hidden_out
        blocks.append(self.discriminator_final(hidden_in))
        return nn.Sequential(*blocks)

    def discriminator_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, self.kernel_size, self.stride, self.padding
            ),
            nn.LeakyReLU(0.2, True),
        )

    def discriminator_final(self, in_channels):
        return nn.Conv2d(
            in_channels, self.output_dim, kernel_size=2, stride=1, padding=0
        )

    def forward(self, x_cond):
        return self.blocks(x_cond).view(-1, 1)


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


def train_gan(model, device, data, epochs, lr):
    criterion = nn.BCEWithLogitsLoss()  ##nn.BCELoss()
    g_params = list(model.generator.parameters()) + list(
        model.label_embedding.parameters()
    )
    d_params = list(model.discriminator.parameters()) + list(
        model.label_embedding.parameters()
    )
    generator_optimizer = torch.optim.Adam(
        g_params,
        lr=lr,
    )
    discriminator_optimizer = torch.optim.Adam(
        d_params,
        lr=lr,
    )
    model.train()

    for epoch in range(epochs):
        avg_g_loss = 0.0
        avg_d_loss = 0.0
        for real_images, labels in data:
            real_images = real_images.to(device)
            b_size = real_images.shape[0]
            labels = labels.to(device)
            discriminator_optimizer.zero_grad()
            valid = torch.ones(b_size, 1, device=device)
            fake = torch.zeros(b_size, 1, device=device)

            real_preds = model.discriminate(real_images, labels)
            d_real_loss = criterion(real_preds, valid)
            fake_images = model.generate(labels).detach()
            fake_preds = model.discriminate(fake_images, labels)
            d_fake_loss = criterion(fake_preds, fake)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            fake_images = model.generate(labels)
            fake_preds = model.discriminate(fake_images, labels)
            g_loss = criterion(fake_preds, valid)
            g_loss.backward()
            generator_optimizer.step()

            avg_g_loss += g_loss.item()
            avg_d_loss += d_loss.item()
        print(
            f"[Epoch {epoch+1:03d}] discriminator loss: {avg_d_loss / len(data):.4f} | generator loss: {avg_g_loss / len(data):.4f}"
        )
        model.save("weights/gan.pth")
