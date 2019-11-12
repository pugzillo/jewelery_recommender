import csv
import logging
import multiprocessing
import requests
import shutil

from bs4 import BeautifulSoup

IN_FILEPATH = 'products.csv'
OUT_FILEPATH = 'products_with_img_urls.csv'
OUT_FILEPATH_TMP = f'{OUT_FILEPATH}.tmp'

def scrape_img_url(product_link):
    logging.info(f'Scraping product_link={product_link}...')
    try:
        r = requests.get(f'https://www.shopstyle.com{product_link}')
        soup = BeautifulSoup(r.content, features='html.parser')
        img_url = soup.find('div', {'class': 'slider__main'}) \
            .find('img', {'class': 'product-image'}) \
            ['src']
        return (product_link, img_url)
    except Exception:
        return None


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
)


with open(IN_FILEPATH, 'r') as f:
    reader = csv.DictReader(f, delimiter=',')
    links_without_url = set(row['link'] for row in reader if not row['img_url'].startswith('https://img.shopstyle-cdn.com/sim/'))
logging.info(f'links_without_url={links_without_url}')

with multiprocessing.Pool(multiprocessing.cpu_count() * 4) as workers:
    tpls = workers.map(scrape_img_url, links_without_url)
link_to_url = {tpl[0]: tpl[1] for tpl in tpls if tpl}

with open(IN_FILEPATH, 'r') as in_f, open(OUT_FILEPATH_TMP, 'w') as out_f:
    reader = csv.DictReader(in_f, delimiter=',')
    writer = csv.DictWriter(out_f, delimiter=',', fieldnames=reader.fieldnames)
    if out_f.tell() == 0:
        writer.writeheader()

    for row in reader:
        if row['link'] in link_to_url:
            row['img_url'] = link_to_url[row['link']]
        writer.writerow(row)

shutil.move(OUT_FILEPATH_TMP, OUT_FILEPATH)