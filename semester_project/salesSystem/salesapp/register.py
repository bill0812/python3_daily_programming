from firebase import firebase
import hashlib,re
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def update_to_firebase(name,account,password,sex,phone,address):
    check_account = check_member_register(account)
    if check_account == 1:
        account_key = hashlib.md5(account.encode())
        password_hash = hashlib.md5((password).encode())
        password = password_hash.hexdigest()
        account_key = account_key.hexdigest()
        member = {
                "個人資料" : {
                    "帳號" : account,
                    "密碼" : password,
                    "名字" : name,
                    "性別" : sex,
                    "電話" : phone,
                    "常用取貨地址" : address
                },
                "購物紀錄" : {

                },
                "個人購物車" : {

                }
        }
        fb.put('/會員資料', data = member , name = account_key)
        return 1,account_key,name
    else:
        return 0,"",""

def check_member_register(account):
    acc = hashlib.md5(account.encode())
    member_account = acc.hexdigest()
    member_data = fb.get("/會員資料",None)
    if member_data != None:
        if member_account not in member_data.keys():
            return 1
        else :
            return 0
    else:
        return 0

def fetch_address():
    address_data = fb.get("/各地資訊",None)
    return address_data.keys()

def fetch_district(county):
    district_data = fb.get("/各地資訊/"+county,None)
    return district_data
