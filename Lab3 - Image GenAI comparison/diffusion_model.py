import torch.nn as nn
import torch
import math


class UNet(nn.Module):
    def __init__(
        self,
        input_ch,
        base_ch,
        timestep_emb,
        num_classes,
        label_emb_dim,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.label_embedding = LabelEmbedding(
            num_classes=num_classes, embedding_dim=label_emb_dim
        )
        self.timestep_embedding = SinusoidalTimeEmbedding(timestep_emb)
        self.emb_proj = nn.Sequential(
            nn.Linear(timestep_emb + label_emb_dim, timestep_emb * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb * 4, timestep_emb),
        )
        self.conv_in = nn.Conv2d(
            input_ch, base_ch, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.down1 = ResBlock(
            base_ch,
            base_ch * 2,
            kernel_size=kernel_size,
            emb_dim=timestep_emb,
            stride=stride,
            padding=padding,
        )
        self.down2 = ResBlock(
            base_ch * 2,
            base_ch * 4,
            kernel_size=kernel_size,
            emb_dim=timestep_emb,
            stride=stride,
            padding=padding,
        )
        self.downscale = nn.MaxPool2d(2)

        self.mid_block = ResBlock(
            base_ch * 4,
            base_ch * 4,
            kernel_size=kernel_size,
            emb_dim=timestep_emb,
            stride=stride,
            padding=padding,
        )

        self.up1 = ResBlock(
            base_ch * 8,
            base_ch * 2,
            kernel_size=kernel_size,
            emb_dim=timestep_emb,
            stride=stride,
            padding=padding,
        )
        self.up2 = ResBlock(
            base_ch * 4,
            base_ch,
            kernel_size=kernel_size,
            emb_dim=timestep_emb,
            stride=stride,
            padding=padding,
        )
        self.upscale = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_out = nn.Conv2d(base_ch, input_ch, kernel_size=1, stride=stride)

    def forward(self, x, t, y):
        t_emb = self.timestep_embedding(t)
        y_emb = self.label_embedding(y)
        emb = self.emb_proj(torch.cat([t_emb, y_emb], dim=1))

        h1 = self.conv_in(x)
        h2 = self.down1(h1, emb)
        p1 = self.downscale(h2)
        h3 = self.down2(p1, emb)
        p2 = self.downscale(h3)

        mid = self.mid_block(p2, emb)

        up1 = self.upscale(mid)
        up1 = self.up1(torch.cat([up1, h3], dim=1), emb)
        up2 = self.upscale(up1)
        up2 = self.up2(torch.cat([up2, h2], dim=1), emb)
        return self.conv_out(up2)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        emb_dim,
        stride,
        padding,
        base_num_groups=8,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(base_num_groups, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm2 = nn.GroupNorm(base_num_groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x, emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        skip = self.skip_connection(x) if self.skip_connection is not None else x
        return h + skip


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, y):
        return self.embedding(y)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cuda"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, dim=0)
    return betas, a_bar


def q_sample(x, t, noise, a_bar):
    a = a_bar[t][:, None, None, None]
    return torch.sqrt(a) * x + torch.sqrt(1 - a) * noise


def train_diffusion(model, data, epochs, lr, T=1000, device="cuda"):
    model.train().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    _, a_bar = make_beta_schedule(T, device=device)

    for epoch in range(epochs):
        total = 0
        for x, y in data:
            x, y = x.to(device), y.to(device)

            B = x.shape[0]
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise, a_bar)

            eps_pred = model(x_t, t, y)
            loss = criterion(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"[Epoch {epoch+1:03d}] | loss: {total/len(data):.4f}")
        torch.save(model.state_dict(), "weights/ddpm.pth")


@torch.no_grad()
def generate(
    model, y, device="cuda", T=1000, beta_start=1e-4, beta_end=0.02, clamp=True
):

    betas, a_bar = make_beta_schedule(T, beta_start, beta_end, device)
    a_bar_prev = torch.cat([torch.ones(1, device=device), a_bar[:-1]])

    B = y.shape[0]
    C, H, W = 3, 32, 32
    x_t = torch.randn(B, C, H, W, device=device)

    model.eval().to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps_theta = model(x_t, t_batch, y)

        beta_t = betas[t]
        a_bar_t = a_bar[t]
        a_bar_prev_t = a_bar_prev[t]

        mean = (1.0 / torch.sqrt(1.0 - beta_t)) * (
            x_t - beta_t / torch.sqrt(1.0 - a_bar_t) * eps_theta
        )

        if t > 0:
            var = beta_t * (1.0 - a_bar_prev_t) / (1.0 - a_bar_t)
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(var) * noise
        else:
            x_t = mean

    return x_t.clamp(-1, 1) if clamp else x_t
