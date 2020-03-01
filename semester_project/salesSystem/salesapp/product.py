from firebase import firebase
import hashlib
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def fetch_product():
    product = fb.get("/產品資訊",None)
    if product == None:
        return 1
    else:
        return product

def fetch_product_order(account,customer_status,normal_status):
    customer = dict()
    normal = dict()
    money = int()

    for i in range(len(customer_status)) :
        customer[customer_status[i]] = fb.get("/會員資料/"+account+"/購物車/客製化/"+customer_status[i],None)
        money = money + int(customer[customer_status[i]]["總價格"])
        fb.delete("/會員資料/"+account+"/購物車/客製化/"+customer_status[i],None)
    for i in range(len(normal_status)) :
        normal[normal_status[i]] = fb.get("/會員資料/"+account+"/購物車/產品資訊/"+normal_status[i],None)
        money = money + int(normal[normal_status[i]]["總價格"])
        fb.delete("/會員資料/"+account+"/購物車/產品資訊/"+normal_status[i],None)

    if customer == None and normal == None :
        pass
    elif customer == None or normal == None:
        if customer == None:
            customer = ""
        else:
            normal = ""
    else:
        pass
    return customer,normal,money
