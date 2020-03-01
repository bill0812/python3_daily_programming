from firebase import firebase
import hashlib,re
import salesapp.product as product
import salesapp.customer as customer_data
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def make_order(account,order_time,name,order_name,phone,address,customer_status,normal_status):
    status = customer_data.customer_show()
    if status == 0 :
        product_data = []
    else :
        product_data = status
    order_key = order_time+name
    final_customer,final_normal,money= product.fetch_product_order(account,customer_status,normal_status)
    real_data = {
        "取貨地址" : address,
        "產品資訊" : {
            "客製化" : final_customer,
            "一般商品" : final_normal
        },
        "總金額" : money,
        "訂單成立人" : order_name,
        "電話" : phone
    }
    if final_customer != "" :
        for index,data in final_customer.items():
            for data_name,data_count in data.items():
                if data_name in product_data :
                    print(data_count)
                    #print(fb.get('/蔬菜庫存/'+data_name+"/數量",None)[:-1])
                    final_count = int(fb.get('/蔬菜庫存/'+data_name+"/數量",None)) - int(data_count)
                    fb.put('/蔬菜庫存/'+data_name, data = final_count , name = "數量")

    if final_normal != "" :
        for index,data in final_normal.items():
            for data_name,data_count in data.items():
                if data_name == "數量":
                    count = data_count
                    inventory_count = int(fb.get('/產品資訊/'+index+"/目前存貨",None)) - int(count)
                    fb.put('/產品資訊/'+index, data = inventory_count , name = "目前存貨")
                if data_name == "產品資訊" :
                    for name,quantity in data_count.items() :
                        if name in product_data :
                            final_count = int(fb.get('/蔬菜庫存/'+name+"/數量",None)) - int(count)*int(quantity[:-1])
                            fb.put('/蔬菜庫存/'+name, data = final_count , name = "數量")
    print(final_normal)
    print(real_data)
    fb.put('/會員資料/'+account+"/購物紀錄", data = real_data , name = order_key)
    fb.put('/訂單記錄', data = real_data , name = order_key)
