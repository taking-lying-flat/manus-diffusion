import torch
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os


def visualize_reconstruction(model, dataloader, device, save_path, num_images=8):
    model.eval()
    
    with torch.no_grad():
        inputs, _ = next(iter(dataloader))
        inputs = inputs[:num_images].to(device)
        recon, _, _ = model(inputs)
        
        comparison = torch.cat([inputs, recon])
        save_image(comparison.cpu(), save_path, nrow=num_images, normalize=True, pad_value=1)


def visualize_samples(model, device, save_path, num_samples=64):
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device=device)
        save_image(samples.cpu(), save_path, nrow=8, normalize=True, pad_value=1)


def visualize_interpolation(model, dataloader, device, save_path, num_interpolations=10):
    model.eval()
    
    with torch.no_grad():
        inputs, _ = next(iter(dataloader))
        img1 = inputs[0:1].to(device)
        img2 = inputs[1:2].to(device)
        
        h1 = model.encoder(img1)
        z1, mu1, logvar1 = model._bottleneck(h1)
        
        h2 = model.encoder(img2)
        z2, mu2, logvar2 = model._bottleneck(h2)
        
        interpolations = []
        interpolations.append(img1.cpu())
        
        for alpha in np.linspace(0, 1, num_interpolations):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            z_interp = model.fc3(z_interp)
            interp_img = model.decoder(z_interp)
            interpolations.append(interp_img.cpu())
        
        interpolations.append(img2.cpu())
        
        interpolations = torch.cat(interpolations, dim=0)
        save_image(interpolations, save_path, nrow=num_interpolations+2, normalize=True, pad_value=1)


def visualize_latent_space_2d(model, dataloader, device, save_path, num_samples=1000):
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            if len(latent_vectors) * dataloader.batch_size >= num_samples:
                break
            inputs = inputs.to(device)
            h = model.encoder(inputs)
            _, mu, _ = model._bottleneck(h)
            latent_vectors.append(mu.cpu())
            labels.append(targets)
    
    latent_vectors = torch.cat(latent_vectors, dim=0)[:num_samples].numpy()
    labels = torch.cat(labels, dim=0)[:num_samples].numpy()
    
    if latent_vectors.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization (2D PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_latent_walk(model, device, save_path, dim=0, num_steps=10, range_val=3):
    model.eval()
    
    with torch.no_grad():
        walks = []
        
        for val in np.linspace(-range_val, range_val, num_steps):
            z = torch.zeros(1, model.n_latent_features, device=device)
            z[0, dim] = val
            z = model.fc3(z)
            img = model.decoder(z)
            walks.append(img.cpu())
        
        walks = torch.cat(walks, dim=0)
        save_image(walks, save_path, nrow=num_steps, normalize=True, pad_value=1)


def create_visualization_grid(model, dataloader, device, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    visualize_reconstruction(model, dataloader, device, f"{save_dir}/reconstruction_epoch_{epoch}.png", num_images=8)
    visualize_samples(model, device, f"{save_dir}/samples_epoch_{epoch}.png", num_samples=64)
    visualize_interpolation(model, dataloader, device, f"{save_dir}/interpolation_epoch_{epoch}.png", num_interpolations=10)
    
    try:
        visualize_latent_space_2d(model, dataloader, device, f"{save_dir}/latent_space_epoch_{epoch}.png", num_samples=1000)
    except Exception:
        pass
    
    visualize_latent_walk(model, device, f"{save_dir}/latent_walk_dim0_epoch_{epoch}.png", dim=0, num_steps=10)
