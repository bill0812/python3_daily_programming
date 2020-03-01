if __name__ == "__main__":
    import menu,firebase_data,time,homepage,admin
    while True:
        choice = menu.main_menu()
        if choice == str(1) :
            check_register = ""
            account,password = menu.become_member_first_step()
            name,birthday,sex = menu.become_member_second_step()
            while (check_register != "y" or check_register != "y") :
                check_register = menu.become_member_final_step(account,password,name,birthday,sex)
                member,name = firebase_data.update_to_firebase(account,password,name,birthday,sex)
            homepage.homepage(member,name)
        elif choice == str(2) :
            count = 0
            check_login = 1
            while (check_login != 0 or check_login != 0) and count < 3 :
                account,password = menu.login_menu()
                check_login,member,name = firebase_data.check_member_exsist(account,password)
                count = count + 1
                if member == "NO EXSIST" and name == "NO EXSIST":
                    print("You account isn't exsist !")
                    input("Press any key to leave")
                elif member == "NO Member" and name == "NO Member":
                    print("There's no any member !")
                    input("Press any key to leave")
                else :
                    homepage.homepage(member,name)
            if count >= 3 :
                print("You Enter too many times !")
                input("Press any key to leave")
        elif choice == str(3) :
            check_admin = 0
            while check_admin < 3:
                account,password = menu.admin_menu()
                if account == "0000" and password == "0000":
                    break
                else :
                    pass
                check_admin = check_admin + 1

            if check_admin >=3 :
                print("You Enter too many times !")
                input("Press any key to leave")
            else :
                print("Congrats, Entering ...")
                time.sleep(1)
                admin.admin()
        elif choice == str(4):
            break
        else :
            print("You type the wrong number !")
            input("Press any key to continue ~")
