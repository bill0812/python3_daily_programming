from firebase import firebase
import hashlib,re
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def fetch_phone_address(account):
    phone = fb.get("/會員資料/"+account+"/個人資料/電話",None)
    address = fb.get("/會員資料/"+account+"/個人資料/常用取貨地址",None)
    return phone,address

def fetch_customer_product(account,customer_data):
    all_detail = dict()
    for i in range(len(customer_data)):
        detail = fb.get("/會員資料/"+account+"/購物車/客製化/"+customer_data[i],None)
        all_detail[customer_data[i]] = detail
    return all_detail
def fetch_normal_product(account,normal_data):
    all_detail = dict()
    for i in range(len(normal_data)):
        detail = fb.get("/會員資料/"+account+"/購物車/產品資訊/"+normal_data[i],None)
        all_detail[normal_data[i]] = detail
    return all_detail
