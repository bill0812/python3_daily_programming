import os,firebase_data
def become_member_first_step():
    os.system("clear")
    check_account = 1
    check_pass = "Y"
    x = 0
    y = 0
    while check_account == 1:
        if x == 0 :
            account = input("Enter Your Account (E-mail): ")
        else:
            account = input("Enter Your Account AGAIN !: ")
        check_account = firebase_data.check_account(account)
        x = x +1
    while check_pass == "Y"  :
        if y == 0:
            password = input("Enter Your Password : ")
        else:
            password = input("Enter Your Password AGAIN ! :")
        pass_security = firebase_data.check_pass(password)
        if pass_security == "至少一個大寫":
            print("Your Password's Secuirty : {}".format(pass_security))
            input("Press any key to continue ...")
            check_pass = "Y"
            y = y +1
        else:
            print("Your Password's Secuirty : {}".format(pass_security))
            check_pass = input("Press OK(ok) to leave ...")
            if check_pass == "OK" or check_pass == "oK" or check_pass == "Ok" or check_pass == "ok" :
                check_pass = "N"
            else:
                check_pass = "Y"
                x = 0
    password_confirm = input("Enter Your Password to confirm : ")
    while password != password_confirm:
        password_confirm = input("Enter Your Password Again : ")
    input("Press any key to continue ...")
    return account,password

def become_member_second_step():
    os.system("clear")
    check_name = 0
    check_sex = 0
    x = 0
    y = 0
    while check_name == 0:
        if x == 0 :
            name = input("Enter Your Name : ")
        else :
            name = input("Enter Your Name AGAIN (NOT EMPTY): ")
        if name == "":
            check_name = 0
            x = x + 1
        else:
            check_name = 1
    birthday = input("Enter Your Birthday (YYYY/MM/DD) : ")
    if birthday == "":
        birthday = "empty now"
    while check_sex == 0:
        if y == 0 :
            sex = input("Enter Your Sex (male(1)/female(2)): ")
        else :
            sex = input("Enter Your Sex AGAIN (male(1)/female(2)) !: ")
        if sex == "":
            sex = "empty now"
            check_sex = 1
        elif sex == str(1) or sex == "male" :
            sex = "男性"
            check_sex = 1
        elif sex == str(2) or sex == "female" :
            sex = "女性"
            check_sex = 1
        else :
            y = 1
    input("Press any key to continue ...")
    return name,birthday,sex

def become_member_final_step(account,password,name,birthday,sex):
    os.system("clear")
    print("Check Your Data Again !!! ")
    print("Your Account : {}".format(account))
    print("Your Password : {}".format(password))
    print("Your Name : {}".format(name))
    print("Your Birthday : {}".format(birthday))
    print("Your Sex : {}".format(sex))
    check = input("Press Y(y) for yes ; Press N(n) for no : ")
    return check

def admin_menu():
    os.system("clear")
    print("This is a Admin System :")
    account = input("Admin Acount : ")
    password = input("Admin Password : ")
    input("Press any key to continue ...")
    return account,password

def login_menu():
    os.system("clear")
    print("This is a Login System :")
    account = input("Your Acount : ")
    password = input("Your Password : ")
    input("Press any key to continue ...")
    return account,password

def main_menu():
    os.system("clear")
    print("Press 1 for Register a Member\n")
    print("Press 2 for Login\n")
    print("Press 3 for ADMIN\n")
    print("Press 4 to Leave\n")
    choice = input("Your Choice is : ")
    return choice

def homepage_menu(name):
    os.system("clear")
    print("Welcome {} !! ".format(name))
    print("Press 1 for Buying Normal Product\n")
    print("Press 2 for Customize Your Own Product\n")
    print("Press 3 for Shopping Cart\n")
    print("Press 4 for Setting\n")
    print("Press 5 for Leaving\n")
    choice = input("Your Choice is : ")
    return choice

def admin():
    os.system("clear")
    print("Welcome ADMIN !! ")
    print("Press 1 for Checking Members\n")
    print("Press 2 for Viewing Product\n")
    print("Press 3 for Adding/Deleting Product\n")
    print("Press 4 for Viewing Inventory\n")
    print("Press 5 for Dealing With Inventory\n")
    print("Press 6 for Editing Farmer's Story\n")
    print("Press 7 for Leaving\n")
    choice = input("Your Choice is : ")
    return choice

def store_menu(name):
    os.system("clear")
    product = firebase_data.fetch_product()
    print("Welcome {} !\n".format(name))
    print("Press S(s) to Setting \nPress C(c) to Shopping Cart \nPress L(l) to Leave \n=================================")
    if product == 1:
        print("目前沒有產品哦 ！！")
    else:
        print("目前產品有{}項".format(len(product)))
    choice = input("What are you going to do ? ")
    return choice

def admin_edit_menu_product():
    os.system("clear")
    print("Enter 1 to add product \n")
    print("Enter 2 to delete product \n")
    print("Enter 3 to leave \n")
    choice = input("Enter your choice :")
    return choice

def admin_edit_menu_inventory():
    os.system("clear")
    print("Enter 1 to add inventory \n")
    print("Enter 2 to delete inventory \n")
    print("Enter 3 to revise inventory's amount or calories \n")
    print("Enter 4 to leave \n")
    choice = input("Enter your choice :")
    return choice

def add_product():
    y = "Y"
    product_detail = dict()
    while (y == "Y" or y =="y"):
        os.system("clear")
        product = input("What name of product do you want to add ? ")
        product_detail[product] = dict()
        product_detail[product]["種類"] = dict()
        x = "Y"
        while (x == "Y" or x =="y"):
            vegetable = input("What kind of vegetable do you want to add ? ")
            quantity = input("How many of quantity ? ")
            product_detail[product]["種類"][vegetable] = dict()
            product_detail[product]["種類"][vegetable]["數量"] = quantity
            x = input("Do you want to add any kind of vegetable ? (Y(y) for yes ; N(n) for no )")
            os.system("clear")
        money = input("How much dollars ? ")
        product_detail[product]["價格"] = money
        y = input("Do you want to add any kind of product? (Y(y) for yes ; N(n) for no )")
    firebase_data.add_product(product_detail)
    return product_detail

def add_inventory():
    y = "Y"
    inventory_detail = dict()
    while (y == "Y" or y =="y"):
        os.system("clear")
        inventory = input("What name of product's inventory do you want to add ? ")
        inventory_detail[inventory] = dict()
        quantity = input("數量")
        level = input("進貨物流標準")
        inventory_detail[inventory]["數量"] = quantity
        inventory_detail[inventory]["進貨物流標準"] = level
        y = input("Do you want to add any kind of product's inventory? (Y(y) for yes ; N(n) for no )")
    firebase_data.add_inventory(inventory_detail)
    return inventory_detail

def revise_inventory(inventory_name):
    os.system("clear")
    print("Press 1 to revise {}'s amount ! \n".format(inventory_name))
    print("Press 2 to revise {}'s calories ! \n".format(inventory_name))
    print("Press 3 to leave ! \n")
    choice = input("What is your choice")
    return choice

def admin_edit_menu_story():
    os.system("clear")
    print("Enter 1 to add story \n")
    print("Enter 2 to delete story \n")
    print("Enter 3 to revise story \n")
    print("Enter 4 to leave \n")
    choice = input("Enter your choice :")
    return choice

def add_farmer_story():
    farmer_name = input("Enter Farmer's account : ")
    farmer_story = input("Enter Farmer's Name : ")
    return farmer_name,farmer_story

def revise_story(farmer_name):
    os.system("clear")
    print("Press 1 to revise {}'s name ! \n".format(farmer_name))
    print("Press 2 to revise {}'s story ! \n".format(farmer_name))
    print("Press 3 to leave ! \n")
    choice = input("What is your choice")
    return choice
