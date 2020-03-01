from firebase import firebase
import hashlib
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def check_member_exsist(account,password):
    acc = hashlib.md5(account.encode())
    pas = hashlib.md5(password.encode())
    member_account = acc.hexdigest()
    member_password = pas.hexdigest()
    member_data = fb.get("/會員資料",None)
    print(member_data)
    print(account)
    print(password)
    if member_data != None:
        if member_account not in member_data.keys():
            return 0,"注意","帳號不存在！"
        else :
            member_data =  member_data[member_account]["個人資料"]
            if member_password not in  member_data.values():
                return 0,"注意","錯誤密碼"
            else:
                name = member_data["名字"]
                return 1,member_account,name
    else:
        return 0,"注意","沒有會員資料"

def check_account_exsist(account):
    acc = hashlib.md5(account.encode())
    member_account = acc.hexdigest()
    member_data = fb.get("/會員資料",None)
    if member_data != None:
        if member_account not in member_data.keys():
            return 0
        else :
            return 1
    else:
        return 0
