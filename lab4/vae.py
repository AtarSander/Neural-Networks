import torch.nn as nn
import torch


class VAE(nn.Module):
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
        self.encoder = Encoder(
            input_dim + label_emb_dim,
            hidden_dims,
            latent_dim,
            kernel_size,
            stride,
            padding,
        ).to(device)
        self.decoder = Decoder(
            input_dim,
            hidden_dims,
            latent_dim + label_emb_dim,
            kernel_size,
            stride,
            padding,
        ).to(device)
        self.label_embedding = LabelEmbedding(
            num_classes=num_classes, embedding_dim=label_emb_dim
        ).to(device)
        self.device = device
        self.num_classes = num_classes
        self.latent_dim = latent_dim

    def forward(self, x, y):
        B, C, H, W = x.shape
        y_emb = self.label_embedding(y)
        x_cond = torch.cat([x, y_emb[:, :, None, None].expand(-1, -1, H, W)], dim=1)
        mu, logvar = self.encoder(x_cond)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        z_cond = torch.cat([z, y_emb], dim=1)
        x_hat = self.decoder(z_cond)
        return x_hat, mu, logvar

    def save(self, path):
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "label_embedding": self.label_embedding.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.label_embedding.load_state_dict(ckpt["label_embedding"])

    def generate(self, y):
        with torch.no_grad():
            y_emb = self.label_embedding(y)
            z = torch.randn((y.shape[0], self.latent_dim)).to(self.device)
            z_cond = torch.cat([z, y_emb], dim=1)
            x_gen = self.decoder(z_cond)
        return x_gen.detach().cpu()


class Encoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, latent_dim, kernel_size, stride, padding
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.blocks = self.build_encoder()
        self.mu, self.var = self.distr_params(latent_dim)

    def build_encoder(self):
        blocks = nn.ModuleList()
        hidden_in = self.input_dim
        for hidden_out in self.hidden_dims[:-2]:
            blocks.append(self.encoder_block(hidden_in, hidden_out))
            hidden_in = hidden_out
        blocks.append(self.encoder_final())
        return nn.Sequential(*blocks)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        )

    def encoder_final(self):
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.hidden_dims[-2], self.hidden_dims[-1]),
        )

    def distr_params(self, latent_dim):
        return (
            nn.Linear(self.hidden_dims[-1], latent_dim),
            nn.Linear(self.hidden_dims[-1], latent_dim),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.mu(x), self.var(x)


class Decoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, latent_dim, kernel_size, stride, padding
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims[::-1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.blocks = self.build_decoder(latent_dim)

    def build_decoder(self, latent_dim):
        blocks = nn.ModuleList()
        blocks.append(self.decoder_input(latent_dim))
        hidden_in = self.hidden_dims[2]
        for hidden_out in self.hidden_dims[3:]:
            blocks.append(self.decoder_block(hidden_in, hidden_out))
            hidden_in = hidden_out
        blocks.append(self.decoder_final(hidden_in, self.input_dim))
        return nn.Sequential(*blocks)

    def decoder_input(self, latent_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.Unflatten(1, (self.hidden_dims[2], self.kernel_size, self.kernel_size)),
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        )

    def decoder_final(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.blocks(x)


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, y):
        return self.embedding(y)


def train_vae(model, device, data, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="sum")
    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        for img_real, label_real in data:
            img_real = img_real.to(device)
            label_real = label_real.to(device)
            img_gen, mu, var = model(img_real, label_real)
            reconstruction_loss = criterion(img_gen, img_real)
            kl_divergence = -0.5 * torch.sum(1 + var - mu**2 - var.exp())
            total_loss = reconstruction_loss + kl_divergence
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += total_loss.item()
        print(f"[Epoch {epoch+1:03d}] | loss: {avg_loss/len(data):.4f}")
        model.save(f"weights/vae.pth")


def initialize_weights(model):
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
