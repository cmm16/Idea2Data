import src.data_pipeline as dp
import src.data as data
import torch.utils.data
from src.variational_auto_encoder import VAE
from src.compile_vae import train_vae

dp.GoogleImageCrawler()
dp.standardize_image_types()
data_set = data.ScrapeImageDataset()
dataloader = torch.utils.data.DataLoader(data_set, batch_size=bs, shuffle=True)
vae = VAE()
recon_images = train_vae()
data_set.remove_outliers(recon_images)
if generate:
    data_set.generate_new_images()
