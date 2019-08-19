import torch
from src.variational_auto_encoder import VAE
import torch.nn.functional as F


def train_vae(model, data_loader, batch_size=32, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model.load_state_dict(torch.load('vae.torch', map_location='cpu'))
    except FileNotFoundError:
        vae = model
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for idx, (images) in enumerate(data_loader):
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1,
                                                                        epochs, loss.data.item() / batch_size,
                                                                        bce.data.item() / batch_size, kld.data.item() / batch_size)
            print(to_print)

    return recon_images


def loss_fn(x_reconstructed, x, mu, logvar):
    print(x_reconstructed.shape, x.shape)
    BCE = F.binary_cross_entropy(x_reconstructed, x, size_average=False)
    # MSE = F.mse_loss(recon_x, x, size_average=False)

    # VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD







