import menu
def homepage(member_login,name_login):
    global member,name
    member = member_login
    name = name_login
    while True:
        choice = menu.homepage_menu(name)
        print(type(choice))
        print(choice)
        print(str(1))
        if choice == str(1):
            choice = menu.store_menu(name)
        elif choice == str(2):
            menu.customer_menu(name)
        elif choice == str(3):
            menu.shoppingcart_menu(name)
        elif choice == str(4):
            menu.setting_menu(name)
        elif choice == str(5):
            break
        else:
            print("You type the wrong number !")
            input("Press any key to continue ~")
