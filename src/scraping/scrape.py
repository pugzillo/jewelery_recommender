"""
47k bracelets
109k earrings 156k
88k necklaces 244k
61k rings 305k

305k * 100KB -> ~30GB of images
"""

from bs4 import BeautifulSoup
import csv
import json
import logging
import multiprocessing
import requests
import shutil

BRANDS_PATH = 'brands.csv'
SCRAPED_BRANDS_PATH = 'scraped_brands.csv'
COLORS = [
    'Beige',
    'Black',
    'Blue',
    'Brown',
    'Gold',
    'Gray',
    'Green',
    'Orange',
    'Pink',
    'Purple',
    'Red',
    'Silver',
    'White',
    'Yellow',
]
JEWELRY_TYPES = [
    'bracelets',
    'earrings',
    'necklaces',
    'rings',
]
PRODUCTS_FILEPATH = 'products.csv'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
)


def get_brands(brands_path, scraped_brands_path):
    scraped_brands = set()
    with open(scraped_brands_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        scraped_brands = set(row['brand'] for row in reader)

    with open(brands_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        return [row['brand'] for row in reader if row['brand'] not in scraped_brands]


def scrape(brand, jewelry_type, color):
    r = requests.get(
        f'https://www.shopstyle.com/browse/{jewelry_type}/{brand}?c={color}')
    soup = BeautifulSoup(r.content, features='html.parser')
    script = soup.find('script', {'id': 'app-state'})
    if not script:
        logging.error(f'no script found for jewelry_type={jewelry_type}&color={color}')
        return
    products = json.loads(script.text.replace('&q;', '"'))[
        'siteSearch']['products']
    with open(PRODUCTS_FILEPATH, 'a') as f:
        f.write(json.dumps(products) + '\n')


def scrape_brand(brand):
    logging.info(f'At brand={brand}...')
    for jewelry_type in JEWELRY_TYPES:
        for color in COLORS:
            scrape(brand, jewelry_type, color)


# At multiprocessing.cpu_count() * 2, scraped 24 in 8min.
# At multiprocessing.cpu_count() * 4, scraped 45 in 15min,
# then in 2nd run 3 in 10min (maybe I got rate-limited, maybe my internet started sucking),
# then in 3rd run 27 in 30min.
# 546brands -> ~9hr
with multiprocessing.Pool(multiprocessing.cpu_count() * 4) as workers:
    workers.map(scrape_brand, get_brands(BRANDS_PATH, SCRAPED_BRANDS_PATH))