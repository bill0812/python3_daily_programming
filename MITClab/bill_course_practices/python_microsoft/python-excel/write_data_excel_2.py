from openpyxl import Workbook

filename = "hello_world.xlsx"

workbook = Workbook()
sheet = workbook.active

sheet["A1"] = "hello"
sheet["B1"] = "world!"

workbook.save(filename=filename)

def print_rows():
    for row in sheet.iter_rows(values_only=True):
        print(row)

sheet["A1"] = "value"
cell = sheet["A1"]
print(cell.value)

# Before, our spreadsheet has only 1 row
print_rows()

# Try adding a value to row 10
sheet["B10"] = "test"
print_rows()

# Insert a column before the existing column 1 ("A")
sheet.insert_cols(idx=1)
print_rows()

# Insert 5 columns between column 2 ("B") and 3 ("C")
sheet.insert_cols(idx=3, amount=5)
print_rows()

# Delete the created columns
sheet.delete_cols(idx=3, amount=5)
sheet.delete_cols(idx=1)
print_rows()

# Insert a new row in the beginning
sheet.insert_rows(idx=1)
print_rows()

# Insert 3 new rows in the beginning
sheet.insert_rows(idx=1, amount=3)
print_rows()

# Delete the first 4 rows
sheet.delete_rows(idx=1, amount=4)
print_rows()