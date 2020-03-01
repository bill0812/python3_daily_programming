from firebase import firebase
import hashlib,re
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def fetch_cart(account):
    customer_data = fb.get("/會員資料/"+account+"/購物車/客製化",None)
    product_data = fb.get("/會員資料/"+account+"/購物車/產品資訊",None)
    return customer_data,product_data

def upload_normal(name,count,account):
    data = dict()
    product_data = fb.get("/會員資料/"+account+"/購物車/產品資訊/"+name,None)
    if product_data == None:
        product_detail = fb.get("/產品資訊/"+name,None)
        data = {
            "數量" : count,
            "總價格" : int(product_detail["價格"]) * int(count),
            "產品資訊" : product_detail["種類"]
        }
        fb.put("/會員資料/"+account+"/購物車/產品資訊/", data = data , name = name)
    else:
        product_detail = fb.get("/產品資訊/"+name,None)
        data = {
            "數量" : int(product_data["數量"]) + int(count),
            "總價格" : int(product_data["總價格"]) + int(product_detail["價格"]) * int(count),
            "產品資訊" : product_detail["種類"]
        }
        fb.put("/會員資料/"+account+"/購物車/產品資訊/", data = data , name = name)
