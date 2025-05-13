import torch.nn as nn
import torch

class GAN(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, num_classes, label_emb_dim, device, input_shape, generator_batchnorm=False):
        super().__init__()
        self.generator = Generator(latent_dim + label_emb_dim, hidden_dims, input_dim, input_shape, generator_batchnorm).to(device)
        self.discriminator = Discriminator(input_dim + label_emb_dim, hidden_dims, 1).to(device)
        self.label_embedding = LabelEmbedding(num_classes=num_classes, embedding_dim=label_emb_dim).to(device)
        self.device = device
        self.num_classes = num_classes

        self.latent_dim = latent_dim
    
    def save(self, path):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'label_embedding': self.label_embedding.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt['generator'])
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.label_embedding.load_state_dict(ckpt['label_embedding'])
    
    def generate(self, y): # y should already be .to(device), is it longtensor?
        z = torch.randn((y.shape[0], self.latent_dim), device=self.device)
        y_emb = self.label_embedding(y)
        z_cond = torch.cat([z, y_emb], dim=1)
        return self.generator(z_cond)
    
    def discriminate(self, x, y): # y should already be .to(device), is it longtensor?
        y_emb = self.label_embedding(y)
        x = x.view(x.size(0), -1)
        x_cond = torch.cat([x, y_emb], dim=1)
        return self.discriminator(x_cond)


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, input_shape, generator_batchnorm=False):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.input_shape = input_shape
        self.blocks = self.build_generator(generator_batchnorm)

    def build_generator(self, generator_batchnorm):
        blocks = nn.ModuleList()
        hidden_in = self.input_dim
        for hidden_out in self.hidden_dims[:-1]:
            blocks.append(self.generator_block(hidden_in, hidden_out, generator_batchnorm))
            hidden_in = hidden_out
        blocks.append(self.generator_final())
        return nn.Sequential(*blocks)
    
    def generator_block(self, in_channels, out_channels, generator_batchnorm):
        if generator_batchnorm:
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LeakyReLU(0.2)
            )
    
    def generator_final(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.output_dim),
            nn.Tanh()
        )

    def forward(self, z_cond):
        return self.blocks(z_cond).view(-1, self.input_shape)
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.blocks = self.build_discriminator()

    def build_discriminator(self):
        blocks = nn.ModuleList()
        hidden_in = self.input_dim
        for hidden_out in self.hidden_dims[:-1]:
            blocks.append(self.discriminator_block(hidden_in, hidden_out))
            hidden_in = hidden_out
        blocks.append(self.discriminator_final())
        return nn.Sequential(*blocks)
    
    def discriminator_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def discriminator_final(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x_cond):
        return self.blocks(x_cond)

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

def train_gan(model, device, data, epochs, lr):
    criterion = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        avg_g_loss = 0.0
        avg_d_loss = 0.0
        for real_images, labels in data:
            b_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # === Train Discriminator ===
            discriminator_optimizer.zero_grad()

            # Real labels
            valid = torch.ones(b_size, 1, device=device)
            fake = torch.zeros(b_size, 1, device=device)

            # Real loss
            real_preds = model.discriminate(real_images, labels)
            d_real_loss = criterion(real_preds, valid)

            # Fake data
            fake_images = model.generate(labels).detach()
            fake_preds = model.discriminate(fake_images, labels)
            d_fake_loss = criterion(fake_preds, fake)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()

            # === Train Generator ===
            generator_optimizer.zero_grad()

            fake_images = model.generate(labels)
            fake_preds = model.discriminate(fake_images, labels)
            g_loss = criterion(fake_preds, valid)

            g_loss.backward()
            generator_optimizer.step()

            avg_g_loss += g_loss.item()
            avg_d_loss += d_loss.item()

        print(f"[Epoch {epoch+1}] discriminator loss: {avg_g_loss / len(data)} | generator loss: {avg_g_loss / len(data)}")
        model.save("weights/gan.pth")