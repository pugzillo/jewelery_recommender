import collections
import csv
import logging
import json
import multiprocessing
import pathlib
import requests
import shutil



PRODUCTS_FILEPATH = 'products.csv'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
)


id_to_url = {}

with open(PRODUCTS_FILEPATH, 'r') as f:
    for i, row in enumerate(f):
        if i % 500 == 0:
            print(i)
        for product in json.loads(row):
            id_to_url[product['id']] = product['image']['sizes']['Best']['url']


def scrape_jpg(tpl):
    id, url = tpl
    logging.info(f'scraping id={id}...')
    r = requests.get(url, stream=True)
    with open(
        f'data/{id}.jpg',
        'wb',
    ) as f:
        shutil.copyfileobj(r.raw, f)


pathlib.Path('data').mkdir(exist_ok=True)

with multiprocessing.Pool(multiprocessing.cpu_count() * 4) as workers:
    workers.map(scrape_jpg, id_to_url.items())