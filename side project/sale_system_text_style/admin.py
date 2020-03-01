import menu,os,firebase_data
def admin():
    global name
    name = "ADMIN"
    while True:
        choice = menu.admin()
        if choice == str(1):
            break
        elif choice == str(2):
            product = firebase_data.fetch_product()
            os.system("clear")
            if product == 1:
                print("There's No Product Info")
                input("Press any key to continue ~")
            else:
                for key,value in product.items():
                    print("You have entered {} .\n".format(key))
                    for value_key,value_value in value.items():
                        print("You have entered {} .\n".format(value_key))
                        for value_value_key,value_value_value in value_value.items():
                            print("==>{}:{}\n".format(value_value_key,value_value_value))
                        print("==========================================")
                        input("Press any key to continue ~")
        elif choice == str(3):
            while True:
                choice_edit = menu.admin_edit_menu_product()
                if choice_edit == "1":
                    product_detail = menu.add_product()
                    os.system("clear")
                    for key,value in product_detail.items():
                        print("You have entered {} .\n".format(key))
                        for length in range(len(value.keys())):
                            if length == 0:
                                for value_key,value_value in value["種類"].items():
                                    print("You have entered {} .\n".format(value_key))
                                    for value_value_key,value_value_value in value_value.items():
                                        print("==>{}:{}\n".format(value_value_key,value_value_value))
                                    print("==========================================")
                            else:
                                print("The Price is {} .\n".format(value["價格"]))
                                print("==========================================")
                    input("Press any key to continue ~")
                elif choice_edit == "2":
                    product = firebase_data.fetch_product()
                    product_name = list()
                    count = 1
                    if product == 1:
                        print("There's No Product Info")
                        input("Press any key to continue ~")
                    else:
                        for key,value in product.items():
                            print("{} : You have {} .\n".format(count,key))
                            product_name.append(key)
                            count = count + 1
                        choice = input("Which Product do you want to delete ? ")
                        firebase_data.delete_product(product_name[int(choice)-1])
                        print("Deletion of {} Complete".format(product_name[int(choice)-1]))
                        input("Press any key to continue ~")
                elif choice_edit == "3":
                    break
                else:
                    print("You type the wrong number !")
                    input("Press any key to continue ~")
        elif choice == str(4):
            inventory = firebase_data.fetch_inventroy()
            if inventory ==1:
                print("There's No Inventory")
                input("Press any key to continue ~")
            else:
                print(inventory)
                input("Press any key to continue ~")
        elif choice == str(5):
            while True:
                choice_revise = menu.admin_edit_menu_inventory()
                if choice_edit == "1":
                    inventory_detail = menu.add_inventory()
                    os.system("clear")
                    for key,value in inventory_detail.items():
                        print("You have entered {} .\n".format(key))
                        for value_key,value_value in value.items():
                            print("==>{} : {} .\n".format(value_key,value_value))
                        print("==========================================")
                    input("Press any key to continue ~")
                    break
                elif choice_edit == "2":
                    inventory = firebase_data.fetch_inventroy()
                    inventory_name = list()
                    count = 1
                    if inventory == 1:
                        print("There's No Product Info")
                        input("Press any key to continue ~")
                    else:
                        for key,value in inventory.items():
                            print("{} : You have {} .\n".format(count,key))
                            product_name.append(key)
                            count = count + 1
                        choice = input("Which Product do you want to delete ? ")
                        firebase_data.delete_inventory(product_name[int(choice)-1])
                        print("Deletion of {} Complete".format(product_name[int(choice)-1]))
                        input("Press any key to continue ~")
                    break
                elif choice_edit == "3":
                    inventory = firebase_data.fetch_inventroy()
                    product_name = list()
                    count = 1
                    if inventory == 1:
                        print("There's No Product Info")
                        input("Press any key to continue ~")
                    else:
                        for key,value in inventory.items():
                            print("{} : You have {} .\n".format(count,key))
                            product_name.append(key)
                            count = count + 1
                        choice = input("Which Product do you want to revise ? ")
                        choice_revise = menu.revise_inventory(product_name[int(choice)-1])
                        while True:
                            if choice_revise == 1:
                                firebase_data.revise_inventory(inventroy_name,1)
                                break
                            elif choice_revise ==2:
                                firebase_data.revise_inventory(inventroy_name,2)
                                break
                            elif choice_revise == 3:
                                break
                            else:
                                print("You type the wrong number !")
                                input("Press any key to continue ~")
                        print("Revise of {} Complete".format(product_name[int(choice)-1]))
                        input("Press any key to continue ~")
                        break
                elif choice_edit == "4":
                    break;
                else:
                    print("You type the wrong number !")
                    input("Press any key to continue ~")
        elif choice == str(6):
            while True:
                choice = menu.admin_edit_menu_story()
                if choice == 1:
                    name,story = menu.add_farmer_story()
                    add_farmer_story(name,story)
                    print("Succeed Adding Story !")
                    choice = input("Press any key to continue :")
                    break
                elif choice == 2:
                    farmer = firebase_data.fetch_farmer()
                    count = 1
                    if farmer == 1:
                        print("There's No Farmer Info")
                        input("Press any key to continue ~")
                    else:
                        for name in farmer:
                            print("{} : You have {} .\n".format(count,key))
                            count = count + 1
                        choice = input("Which Farmer do you want to delete ? ")
                        result = firebase_data.delete_farmer(farmer[int(choice)-1])
                        if result == "NO EXSIST":
                            print("There's no this member !")
                            input("Press any key to continue ~")
                        else:
                            print("Deletion of {} Complete".format(farmer[int(choice)-1]))
                            input("Press any key to continue ~")
                    break
                elif choice == 3:
                    choice_revise = menu.revise_story()
                    farmer = firebase_data.fetch_farmer()
                    count = 1
                    if farmer == 1:
                        print("There's No Farmer Info")
                        input("Press any key to continue ~")
                    else:
                        for name in farmer:
                            print("{} : You have {} .\n".format(count,key))
                            count = count + 1
                        choice = input("Which Farmer do you want to revise ? ")
                        choice_revise = menu.revise_story(farmer[int(choice)-1])
                        while True:
                            if choice_revise == 1:
                                result = firebase_data.revise_story(inventroy_name,1)
                                if result == 1 :
                                    print("No Exsist")
                                else:
                                    print("Succeed Revise !")
                                    input("Press any key to continue")
                                break
                            elif choice_revise ==2:
                                result = firebase_data.revise_story(inventroy_name,2)
                                if result == 1 :
                                    print("No Exsist")
                                else:
                                    print("Succeed Revise !")
                                    input("Press any key to continue")
                                break
                            elif choice_revise == 3:
                                break
                            else:
                                print("You type the wrong number !")
                                input("Press any key to continue ~")
                        print("Revise of {} Complete".format(product_name[int(choice)-1]))
                        input("Press any key to continue ~")
                        break
                elif choice_edit == "4":
                    break;
                else:
                    print("You type the wrong number !")
                    input("Press any key to continue ~")
        elif choice == str(7):
            break
        else:
            print("You type the wrong number !")
            input("Press any key to continue ~")
