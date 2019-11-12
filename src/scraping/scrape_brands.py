from bs4 import BeautifulSoup
import csv
import logging
import requests

logging.basicConfig(level=logging.DEBUG)


brands = []

def get_brands(q):
    logging.info(f'Entered get_brands({q})...')
    r = requests.get(f'https://www.shopstyle.com/brands/earrings?q={q}')
    soup = BeautifulSoup(r.content, features='html.parser')
    brands = []
    for brand in soup.find_all(
            'div',
            {'class': 'brand-category__brand'},
        ):
        res = brand.get_text().strip()
        brands.append(res)
    return brands

for q in range(ord('a'), ord('z') + 1):
    brands.extend(get_brands(chr(q)))
brands.extend(get_brands('0'))


with open('brands.csv', 'w') as f:
    writer = csv.DictWriter(f, delimiter=',', fieldnames=['brand'])
    writer.writeheader()
    for b in brands:
        writer.writerow({'brand': b})

logging.info('scrape_brands.py done')