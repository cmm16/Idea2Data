from os import listdir, mkdir, rename, remove
from os.path import isfile, join
from tqdm import tqdm
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import imghdr
from PIL import Image
from PIL import ImageOps
from icrawler.builtin import GoogleImageCrawler


def google_image_scraper(save_dir: str, keyword: str, max_num: int):
    """Function uses a google image crawler to scrape a desired number of google images and save to a directory"""
    assert isinstance(save_dir, str)
    assert isinstance(keyword, str)
    assert isinstance(max_num, int)
    google_crawler: GoogleImageCrawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    google_crawler.crawl(keyword=keyword, max_num=max_num)


def standardize_image_types(image_directory, img_size, grey_scale = False):
    img_list = listdir(image_directory)
    for img in tqdm(img_list):
        image = join(image_directory, img)
        image_present = True
        img_type = imghdr.what(image)

        if img_type == "webp":
            remove(image)
            image_present = False
        if img_type is None:
            try:
                drawing = svg2rlg(image)

                try:
                    renderPM.drawToFile(drawing, image, fmt="PNG")
                    remove(image)

                except ValueError:
                    remove(image)
                    image_present = False

                except AttributeError:
                    remove(image)
                    image_present = False
            except:
                remove(image)
                image_present = False
        if image_present:
            to_thumbnail(image, img_size)


def to_thumbnail(img_file, img_dims):
    try:
        im = Image.open(img_file)
        remove(img_file)
        new_im = ImageOps.fit(im, img_dims)
        new_im.save(img_file, "JPEG")
    except IOError:
        remove(img_file)





