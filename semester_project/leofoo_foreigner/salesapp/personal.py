from firebase import firebase
import hashlib,re
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def fetch_person(account):
    personal_data = fb.get("/會員資料/"+account+"/個人資料",None)
    print(personal_data)
    order_data = fb.get("/會員資料/"+account+"/購物紀錄",None)
    return personal_data,order_data

def edit_data(account,name,kind):
    fb.put('/會員資料/'+account+"/個人資料/", data = name , name = kind)
