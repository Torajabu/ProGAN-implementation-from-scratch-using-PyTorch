import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
from tqdm import tqdm

# Configuration parameters
Z_DIM = 512
IN_CHANNELS = 256
CHANNELS_IMG = 3
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESSIVE_EPOCHS = [5, 5, 5]  # epochs for 4x4, 8x8, 16x16
START_TRAIN_AT_IMG_SIZE = 4
LAMBDA_GP = 10
BATCH_SIZE = 64
MAX_RESOLUTION_STEP = 2  # Maximum resolution step (16x16)

# Define Generator model
class Generator(nn.Module):
    def __init__(self, z_dim, init_channels, img_channels, max_resolution_step=2):
        super().__init__()
        self.z_dim = z_dim
        self.init_channels = init_channels
        self.img_channels = img_channels
        
        # Initial 4x4 block
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, init_channels, 4, 1, 0),
            nn.LeakyReLU(0.2)
        )
        
        # RGB output for 4x4 resolution
        self.rgb_layers = nn.ModuleList([
            nn.Conv2d(init_channels, img_channels, 1, 1, 0)
        ])
        
        # Progressive blocks
        self.prog_blocks = nn.ModuleList()
        
        # Channel sizes for each resolution
        in_channels = init_channels
        for i in range(max_resolution_step):
            out_channels = in_channels // 2
            self.prog_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2)
                )
            )
            self.rgb_layers.append(
                nn.Conv2d(out_channels, img_channels, 1, 1, 0)
            )
            in_channels = out_channels
    
    def fade_in(self, alpha, upsampled, generated):
        # Apply tanh after weighted sum for proper normalization
        return torch.tanh(alpha * generated + (1 - alpha) * upsampled)
    
    def forward(self, x, alpha, step):
        # Initial 4x4 block
        out = self.initial(x)
        
        # Handle base case (4x4 resolution)
        if step == 0:
            return torch.tanh(self.rgb_layers[0](out))
        
        # Go through required blocks
        for i in range(step):
            if i == step - 1:
                # Save output for fade-in
                prev_rgb = self.rgb_layers[i](out)
                # Process through the last block
                out = self.prog_blocks[i](out)
                # Get the RGB for the higher resolution
                curr_rgb = self.rgb_layers[i+1](out)
                # Fade in the higher resolution
                upsampled = nn.Upsample(scale_factor=2, mode='nearest')(prev_rgb)
                return self.fade_in(alpha, upsampled, curr_rgb)
            else:
                out = self.prog_blocks[i](out)
        
        # This should never be reached
        return None


# Define Discriminator model with properly matching channels
class Discriminator(nn.Module):
    def __init__(self, init_channels, img_channels, max_resolution_step=2):
        super().__init__()
        
        # Calculate channel sizes for each resolution
        # From 4x4 (highest channel count) to larger resolutions (decreasing channels)
        channels = [init_channels]
        for i in range(max_resolution_step):
            channels.append(channels[-1] // 2)
        
        # Reverse the list so channels[0] is for highest resolution (16x16)
        # and channels[-1] is for 4x4
        channels = channels[::-1]
        
        # For debugging
        print(f"Discriminator channel structure: {channels}")
        
        # RGB layers for each resolution
        self.rgb_layers = nn.ModuleList()
        for channel in channels:
            self.rgb_layers.append(
                nn.Conv2d(img_channels, channel, 1, 1, 0)
            )
        
        # Progressive blocks (downsampling)
        self.prog_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            # From higher resolution to lower (e.g., 16x16 -> 8x8 -> 4x4)
            self.prog_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(channels[i], channels[i+1], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2)
                )
            )
        
        # Final block (4x4 -> 1)
        self.final_block = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(channels[-1] * 4 * 4, 1)
        )
    
    def fade_in(self, alpha, downsampled, out):
        return alpha * out + (1 - alpha) * downsampled
    
    def forward(self, x, alpha, step):
        # Step 0: 4x4, Step 1: 8x8, Step 2: 16x16
        curr_step = MAX_RESOLUTION_STEP - step
        
        # For base case (4x4 resolution)
        if step == 0:
            out = self.rgb_layers[-1](x)  # Use the last RGB layer (4x4)
            return self.final_block(out)
        
        # Get output from current resolution's RGB layer
        out = self.rgb_layers[curr_step](x)
        
        if alpha < 1:
            # For fade-in, get downsampled input and process it
            downsampled = nn.AvgPool2d(2)(x)
            downsampled_rgb = self.rgb_layers[curr_step + 1](downsampled)
            
            # Process current resolution
            current_out = self.prog_blocks[curr_step](out)
            
            # Apply fade-in
            out = self.fade_in(alpha, downsampled_rgb, current_out)
            
            # Process through remaining blocks
            curr_step += 1
        else:
            # Alpha = 1, no fade-in needed
            out = self.prog_blocks[curr_step](out)
        
        # Process through remaining blocks to 4x4
        for i in range(curr_step, MAX_RESOLUTION_STEP):
            out = self.prog_blocks[i](out)
        
        # Final block
        return self.final_block(out)


def train_one_epoch(
    generator, 
    discriminator, 
    data_loader, 
    dataset_size, 
    resolution_step, 
    fade_alpha, 
    disc_optimizer, 
    gen_optimizer,
    device="cuda",
    lambda_gp=10,
    z_dimension=512
):
    """Train generator and discriminator for one epoch"""
    progress_bar = tqdm(data_loader, leave=True)
    
    for batch_idx, (real_images, _) in enumerate(progress_bar):
        current_batch_size = real_images.shape[0]
        real_images = real_images.to(device)
        
        # Train discriminator
        noise = torch.randn(current_batch_size, z_dimension, 1, 1).to(device)
        fake_images = generator(noise, fade_alpha, resolution_step)
        
        disc_real_output = discriminator(real_images, fade_alpha, resolution_step)
        disc_fake_output = discriminator(fake_images.detach(), fade_alpha, resolution_step)
        
        # Calculate gradient penalty
        gp = compute_gradient_penalty(
            discriminator, real_images, fake_images, fade_alpha, resolution_step, device
        )
        
        # Discriminator loss (WGAN-GP)
        disc_loss = (
            -(torch.mean(disc_real_output) - torch.mean(disc_fake_output))
            + lambda_gp * gp
            + (0.001 * torch.mean(disc_real_output ** 2))  # Drift penalty
        )
        
        # Update discriminator
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        
        # Train generator
        noise = torch.randn(current_batch_size, z_dimension, 1, 1).to(device)
        fake_images = generator(noise, fade_alpha, resolution_step)
        disc_fake_for_gen = discriminator(fake_images, fade_alpha, resolution_step)
        
        # Generator loss (WGAN)
        gen_loss = -torch.mean(disc_fake_for_gen)
        
        # Update generator
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        
        # Update alpha for progressive growing
        fade_alpha += current_batch_size / (
            (PROGRESSIVE_EPOCHS[resolution_step] * 0.5) * dataset_size
        )
        fade_alpha = min(fade_alpha, 1)
        
        # Update progress bar
        progress_bar.set_postfix(
            d_loss=disc_loss.item(),
            g_loss=gen_loss.item(),
            gp=gp.item(),
            alpha=fade_alpha
        )
    
    return fade_alpha


def compute_gradient_penalty(discriminator, real_imgs, fake_imgs, alpha, step, device):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_imgs.size(0)
    
    # Create random interpolation
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated_images = epsilon * real_imgs + (1 - epsilon) * fake_imgs
    interpolated_images.requires_grad_(True)
    
    # Calculate discriminator output on interpolated images
    mixed_scores = discriminator(interpolated_images, alpha, step)
    
    # Take gradients
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty


def save_sample_images(generator, step, num_samples=16, device="cuda", z_dim=512):
    """Generate and save sample images at current resolution step"""
    image_size = 4 * 2 ** step
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
        fake_images = generator(noise, alpha=1.0, step=step)
    
    # Save images
    save_dir = f"generated_samples/step_{step}_{image_size}x{image_size}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create grid of sample images
    save_image(
        fake_images, 
        f"{save_dir}/grid.png", 
        normalize=True, 
        nrow=4
    )
    
    generator.train()


def get_data_loader(img_size, batch_size=32):
    """Get data loader for specified image size using CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Use CIFAR-10 as our dataset
    dataset = datasets.CIFAR10(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader, dataset


def main():
    print(f"Using device: {DEVICE}")
    
    # Initialize models
    generator = Generator(
        z_dim=Z_DIM, 
        init_channels=IN_CHANNELS, 
        img_channels=CHANNELS_IMG,
        max_resolution_step=MAX_RESOLUTION_STEP
    ).to(DEVICE)
    
    discriminator = Discriminator(
        init_channels=IN_CHANNELS, 
        img_channels=CHANNELS_IMG,
        max_resolution_step=MAX_RESOLUTION_STEP
    ).to(DEVICE)
    
    # Initialize optimizers
    gen_optimizer = optim.Adam(
        generator.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.0, 0.99)
    )
    disc_optimizer = optim.Adam(
        discriminator.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.0, 0.99)
    )
    
    # Set models to training mode
    generator.train()
    discriminator.train()
    
    # Create directory for generated images
    os.makedirs("generated_samples", exist_ok=True)
    
    # Start training from specified resolution step
    start_step = int(math.log2(START_TRAIN_AT_IMG_SIZE / 4))
    
    # Progressive training loop
    for current_step in range(start_step, MAX_RESOLUTION_STEP + 1):
        fade_alpha = 1e-5  # Reset alpha for new resolution
        current_res = 4 * 2 ** current_step
        
        # Get data loader for current resolution
        data_loader, dataset = get_data_loader(img_size=current_res, batch_size=BATCH_SIZE)
        
        print(f"Training at resolution: {current_res}x{current_res}")
        
        # Train for specified number of epochs at current resolution
        for epoch in range(PROGRESSIVE_EPOCHS[current_step]):
            print(f"Epoch [{epoch+1}/{PROGRESSIVE_EPOCHS[current_step]}]")
            
            fade_alpha = train_one_epoch(
                generator=generator,
                discriminator=discriminator,
                data_loader=data_loader,
                dataset_size=len(dataset),
                resolution_step=current_step,
                fade_alpha=fade_alpha,
                disc_optimizer=disc_optimizer,
                gen_optimizer=gen_optimizer,
                device=DEVICE,
                lambda_gp=LAMBDA_GP,
                z_dimension=Z_DIM
            )
        
        # Generate sample images after training at current resolution
        save_sample_images(generator, current_step)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
