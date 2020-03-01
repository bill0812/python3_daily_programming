from firebase import firebase
import hashlib,re
url = "https://sales-system-project.firebaseio.com/"
check_account_re = re.compile("[a-zA-Z0-9_]+@[a-zA-Z0-9\.]+[a-zA-Z]")
check_pass_big = re.compile("[A-Z]+")
check_pass_small = re.compile("[a-z]+")
check_pass_num = re.compile("[0-9]+")
fb = firebase.FirebaseApplication(url, None)
def check_account(account):
    x = 0
    checked = check_account_re.match(account)
    for length in range(len(account)):
        if "@" == account[length]:
            x = x +1
    if checked == None:
        return 1
    else:
        if x > 1:
            return 1
        else:
            return 0

def check_pass(password):
    checked_big = check_pass_big.search(password)
    checked_small = check_pass_small.search(password)
    checked_num = check_pass_num.search(password)
    if checked_small == None:
        checked_small = 0
    else:
        checked_small = len(checked_small.group())
    if checked_num == None :
        checked_num = 0
    else:
        checked_num = len(checked_num.group())
    if checked_big == None:
        return "至少一個大寫"
    elif checked_small == 0 and checked_num  == 0:
        return "長度太短"
    elif (checked_small + checked_num) < 5 :
        return "安全性中而已"
    elif (checked_small + checked_num) < 3 :
        return "不良安全性"
    else:
        return "安全"

def update_to_firebase(account,password,name,birthday,sex):
    key = hashlib.md5((account+password).encode())
    password_hash = hashlib.md5((password).encode())
    password = password_hash.hexdigest()
    member_key = key.hexdigest()
    member = {
            "個人資料" : {
                "帳號" : account,
                "密碼" : password,
                "名字" : name,
                "生日" : birthday,
                "性別" : sex,
            },
            "購物紀錄" : {

            },
            "個人購物車" : {

            }
    }
    fb.put('/會員資料', data = member , name = member_key)
    return member_key,name

def check_member_exsist(account,password):
    key = hashlib.md5((account+password).encode())
    member_key = key.hexdigest()
    member_data = fb.get("/會員資料",None)
    if member_data != None:
        if member_key not in member_data.keys():
            return 1,"NO EXSIST","NO EXSIST"
        else :
            name = member_data[member_key]["個人資料"]["名字"]
            return 0,member_key,name
    else:
        return 0,"NO Member","NO Member"

def fetch_product():
    product = fb.get("/產品資訊",None)
    if product == None:
        return 1
    else:
        return product

def checking_member():
    pass

def add_product(product_detail):
    print(product_detail)
    for key,value in product_detail.items():
        for length in range(len(value.keys())):
            if length == 1:
                for value_key,value_value in value["種類"].items():
                        for value_value_key,value_value_value in value_value.items():
                            fb.put('/產品資訊/'+key+"/種類/"+value_key, data = value_value_value , name = value_value_key)
            else:
                print(value["價格"])
                fb.put('/產品資訊/'+key, data = value["價格"], name = "價格" )

def delete_product(product):
    fb.delete('/產品資訊/'+product,None)

def fetch_inventroy():
    inventory = fb.get("/庫存",None)
    if inventory == None:
        return 1
    else:
        return inventory

def add_inventory(inventory_detail):
    for key,value in inventory_detail.items():
        for value_key,value_value in value.items():
            fb.put('/庫存/'+key, data = value_value , name = value_key)

def delete_inventory(inventory):
    fb.delete('/庫存/'+inventory,None)

def revise_inventory(inventroy_name,choice):
    inventory = fb.get("/庫存/"+inventroy_name,None)
    if choice ==1:
        inventory_amount = inventory["數量"]
        print("Your Inventory amount of {} is {}".format(inventroy_name,inventory_amount))
        add_minus = input("Add Or Minus Or Leave: (Press 1 for add ; Press 2 for minus ; Press 3 to leave) ")
        while True:
            if add_minus == 1:
                revise_amount = input("How Many (Much) ? ")
                inventory_amount = inventory_amount + revise_amount
                break
            elif add_minus ==2:
                revise_amount = input("How Many (Much) ? ")
                if revise_amount > inventory_amount:
                    print("Out of Bound !!")
                    input("Press any key to leave !")
                else:
                    inventory_amount = inventory_amount - revise_amount
                break
            elif add_minus ==3:
                break
            else:
                print("You type the wrong number !")
                input("Press any key to continue ~")
        fb.put('/庫存/'+inventroy_name, data = inventory_amount , name = "數量")
    else:
        inventory_cal = inventory["熱量"]
        print("Your Inventory calories of {} is {}".format(inventroy_name,inventory_cal))
        add_minus = input("Add Or Minus Or Leave: (Press 1 for add ; Press 2 for minus ; Press 3 to leave) ")
        while True:
            if add_minus == 1:
                revise_cal = input("How Many (Much) ? ")
                inventory_cal = inventory_cal + revise_cal
                break
            elif add_minus ==2:
                revise_cal = input("How Many (Much) ? ")
                if revise_cal > inventory_cal:
                    print("Out of Bound !!")
                    input("Press any key to leave !")
                else:
                    inventory_cal = inventory_cal - revise_cal
                break
            elif add_minus ==3:
                break
            else:
                print("You type the wrong number !")
                input("Press any key to continue ~")
        fb.put('/庫存/'+inventroy_name, data = revise_cal , name = "熱量")

def fetch_story():
    farmer = fb.get("/小農介紹",None)
    farmer_name = list()
    if farmer == None:
        return 1
    else:
        for key,value in farmer.items():
            farmer_name.append(value[小農名字])
        return farmer_name

def add_farmer_story(name,story):
    data = {
        "小農名字" : name,
        "小農故事" : story
    }
    fb.put('/小農介紹', data)

def delete_farmer(name):
    farmer = fb.get("/小農介紹",None)
    for key,value in farmer.items():
        if name == value["小農名字"]:
            fb.delete("/小農介紹/"+key,None)
            return farmer_name
    return "NO EXSIST"

def revise_story(inventroy_name,choice):
    farmer = fb.get("/小農介紹",None)
    for key,value in farmer.items():
        if name == value["小農名字"]:
            farmer_key = key
            break
        else:
            key = "NO EXSIST"
    if  farmer_key == "NO EXSIST":
        return 1
    else:
        if choice == 1:
            name_revise = input("Revise name : ")
            fb.put('/小農介紹/'+farmer_key, data = name_revise ,name = "小農名字")
        else:
            story_revise = input("Revise story : ")
            fb.put('/小農介紹/'+farmer_key, data = story_revise ,name = "小農故事")
            return 0
