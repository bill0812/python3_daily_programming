from dataclasses import dataclass
from typing import List

@dataclass
class Sale:
    quantity: int

@dataclass
class Product:
    id: str
    name: str
    sales: List[Sale]

import random

# Ignore these for now. You'll use them in a sec ;)
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference

from db_classes import Product, Sale

products = []

# Let's create 5 products
for idx in range(1, 6):
    sales = []

    # Create 5 months of sales
    for _ in range(5):
        sale = Sale(quantity=random.randrange(5, 100))
        sales.append(sale)

    product = Product(id=str(idx),
                      name="Product %s" % idx,
                      sales=sales)
    products.append(product)

workbook = Workbook()
sheet = workbook.active

# Append column names first
sheet.append(["Product ID", "Product Name", "Month 1",
              "Month 2", "Month 3", "Month 4", "Month 5"])

# Append the data
for product in products:
    data = [product.id, product.name]
    for sale in product.sales:
        data.append(sale.quantity)
    sheet.append(data)

chart = LineChart()
data = Reference(worksheet=sheet,
                 min_row=2,
                 max_row=6,
                 min_col=2,
                 max_col=7)

chart.add_data(data, titles_from_data=True, from_rows=True)
sheet.add_chart(chart, "B8")

cats = Reference(worksheet=sheet,
                 min_row=1,
                 max_row=1,
                 min_col=3,
                 max_col=7)
chart.set_categories(cats)

chart.x_axis.title = "Months"
chart.y_axis.title = "Sales (per unit)"

workbook.save(filename="oop_sample.xlsx")