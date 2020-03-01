from firebase import firebase
import hashlib,re,datetime
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def customer_show():
    customer_data = fb.get("/蔬菜庫存",None)
    if customer_data != None:
        return customer_data
    else:
        return 0
def upload_to_cart(name,account,customer):
    print(account)
    print(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    cart_data = fb.get("/會員資料/"+account+"/購物車/客製化",None)
    print(cart_data)
    if cart_data == None:
        count = 1
        fb.put('/會員資料/'+account+"/購物車/客製化", data = customer , name = "第"+str(count)+"筆")
    else:
        count = len(cart_data)
        print(count)
        fb.put('/會員資料/'+account+"/購物車/客製化", data = customer , name = "第"+str(count+1)+"筆")
    return 1
