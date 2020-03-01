from firebase import firebase
from datetime import timedelta
from datetime import datetime
import hashlib,re,math,time
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)
status = dict()
left = dict()
median = dict()
def fetch_order():
    order = fb.get('/訂單記錄', None)
    return order

def fetch_inventory():
    product = fb.get('/蔬菜庫存', None)
    return product

def fetch_product(id):
    product = fb.get('/產品資訊/'+id, None)
    return product

def compose_product(title,quantity,compose_time,ending_time):
    product = fb.get('/產品資訊/'+title+"/種類", None)
    product_list = list()
    quantity_list = list()
    data = dict()
    for key,value in product.items():
        product_list.append(key)
        quantity_list.append(int(value[:-1]))
        data[key] = int(value[:-1])* int(quantity)
    for i in range(len(product_list)):
        inventory = int(fb.get('/蔬菜庫存/'+product_list[i]+"/數量", None))
        inventory = inventory - int(quantity)*int(quantity_list[i])
        print(int(quantity_list[i]))
        fb.put('/蔬菜庫存/'+product_list[i] , data = str(inventory) , name = "數量")

    data["名稱"] = title
    data["數量"] = quantity
    data["取貨時間"]=ending_time
    fb.put('/主排程/產品訂單' , data = data , name = compose_time)
    return 1

def inventory_situation(today_time):
    inventory_data = fb.get('/蔬菜庫存/', None)
    kinds = list()
    cost = dict()
    sales = dict()
    low = dict()
    high = dict()
    inventory_level = dict()
    next_high = dict()
    next_low = dict()
    global status
    global left
    global median
    for key,value in inventory_data.items():
        kinds.append(key)
    today_time = 4
    if int(today_time) == 4 :
        for i in range(len(kinds)):
            data_before = fb.get('/蔬菜庫存/'+kinds[i]+"/前兩期平均需求", None)
            data_now = fb.get('/蔬菜庫存/'+kinds[i], None)
            this_high = 0
            this_low = 0
            for key,value in data_before.items() :
                if key == "1低" or key == "2低":
                    this_low = this_low + value
                else:
                    this_high = this_high + value
            this_high = math.ceil(float(this_high + data_now["本期平均最高需求"])/3)
            this_low = math.ceil(float(this_low + data_now["本期平均最低需求"])/3)
            for key,value in data_before.items() :
                if key == "2低" :
                    fb.put('/蔬菜庫存/'+kinds[i]+"/前兩期平均需求", data = value ,name = "1低")
                elif key == "2高" :
                    fb.put('/蔬菜庫存/'+kinds[i]+"/前兩期平均需求", data = value ,name = "1高")
                elif key == "1低":
                    fb.put('/蔬菜庫存/'+kinds[i]+"/前兩期平均需求", data = data_now["本期平均最低需求"] ,name = "2低")
                else :
                    fb.put('/蔬菜庫存/'+kinds[i]+"/前兩期平均需求", data = data_now["本期平均最高需求"] ,name = "2高")
            fb.put('/蔬菜庫存/'+kinds[i], data = this_high ,name = "本期平均最高需求")
            fb.put('/蔬菜庫存/'+kinds[i], data = this_low ,name = "本期平均最低需求")

    inventory_data_refresh = fb.get('/蔬菜庫存/', None)
    for key,value in inventory_data_refresh.items():
        cost[key] = value["單筆成本"]
        sales[key] = value["單筆售價"]
        low[key] = value["本期平均最低需求"]
        high[key] = value["本期平均最高需求"]
    for i in range(len(kinds)):
        sl = float((float(sales[kinds[i]]) - float(cost[kinds[i]])) / (float(cost[kinds[i]]) + float(sales[kinds[i]]) - float(cost[kinds[i]])))
        inventory_level[kinds[i]] = math.ceil(low[kinds[i]] + sl * (high[kinds[i]] - low[kinds[i]]))
        median[kinds[i]] =  math.ceil((int(fb.get('/蔬菜庫存/'+kinds[i]+"/本期平均最低需求", None)) + int(fb.get('/蔬菜庫存/'+kinds[i]+"/本期平均最高需求", None)))/2)
        inventory_data = fb.get('/蔬菜庫存/'+kinds[i]+"/數量", None)
        if median[kinds[i]] < inventory_level[kinds[i]] :
            status[kinds[i]] = "過量成本"
            left[kinds[i]] = int((inventory_level[kinds[i]] - median[kinds[i]]) * float(cost[kinds[i]]))
        elif median[kinds[i]] > inventory_level[kinds[i]] :
            status[kinds[i]] = "缺貨成本"
            left[kinds[i]] = int((median[kinds[i]] - inventory_level[kinds[i]] ) * float(float(sales[kinds[i]]) - float(cost[kinds[i]])))
        else:
            status[kinds[i]] = "剛好為存貨成本"
            left[kinds[i]] = 0
    print(inventory_level)
    if int(today_time) == 4 :
        for key,data_level in inventory_level.items():
            compose_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            count_time =  datetime.now() + timedelta(days = 3 )
            ending_time = count_time.strftime("%Y%m%d%H%M%S")
            data = dict()
            data["名稱"] = kinds[i]
            data["數量"] = this_high
            data["取貨時間"] = ending_time
            fb.put('/主排程/蔬菜訂貨' , data = data , name = compose_time)
            time.sleep(1)
            fb.put('/蔬菜庫存/'+key, data = data_level ,name = "數量")
    return status,left,median

def send_inventory_order(name,quantity,compose_time,ending_time):
    data = dict()
    data["名稱"] = name
    data["數量"] = quantity
    data["取貨時間"]=ending_time
    fb.put('/主排程/蔬菜訂貨' , data = data , name = compose_time)
    return 1

def fetch_inbound():
    product_data = fb.get("主排程/產品訂單",None)
    vegetable_data = fb.get("主排程/蔬菜訂貨",None)
    product_status = dict()
    vegetable_status = dict()
    compose_time = time.strftime("%Y%m%d", time.localtime())
    if product_data != None :
        for key,value in product_data.items():
            if value["取貨時間"][0:8] == compose_time:
                product_status[value["名稱"]] = "商品已好"
            else:
                product_status[value["名稱"]] = "商品尚未好"

    if vegetable_data != None :
        for key,value in vegetable_data.items():
            if value["取貨時間"][0:8] == compose_time:
                vegetable_status[value["名稱"]] = "商品已好"
            else:
                vegetable_status[value["名稱"]] = "商品尚未好"
    return product_data,product_status,vegetable_data,vegetable_status

def get_product_order(name):
    product_data = fb.get("主排程/產品訂單/"+name,None)
    recent = fb.get("產品資訊/"+product_data["名稱"]+"/目前存貨",None)
    fb.put("產品資訊/"+product_data["名稱"],data = int(recent) + int(product_data["數量"]) ,name = "目前存貨")
    fb.delete("主排程/產品訂單/"+name,None)
    return 1

def get_veg_order(name):
    veg_data = fb.get("主排程/蔬菜訂貨/"+name,None)
    recent = fb.get("蔬菜庫存/"+veg_data["名稱"]+"/數量",None)
    max_recent = fb.get("蔬菜庫存/"+veg_data["名稱"]+"/本期平均最高需求",None)
    if (int(recent) + int(veg_data["數量"])) > int(max_recent):
        fb.put("蔬菜庫存/"+veg_data["名稱"],data = int(recent) + int(veg_data["數量"]) ,name = "本期平均最高需求")
    fb.put("蔬菜庫存/"+veg_data["名稱"],data = int(recent) + int(veg_data["數量"]) ,name = "數量")
    fb.delete("主排程/蔬菜訂貨/"+name,None)
    return 1

def rfm():
    count_r = 0
    count_f = 0
    count_m = 0
    count_number = 8
    delivery_cost = 1
    per_profit = 5
    lose_balance = 0.02
    level = ["222" , "221" , "212" , "211" , "122" , "121" , "112" , "111"]
    response = [0.3 , 0.2, 0.08, 0.055, 0.03, 0.015, 0.009, 0.0015]
    count_time =  datetime.now() + timedelta(days = 31) - timedelta(days = 45)
    check_time = count_time.strftime("%Y%m%d")
    date_data = fb.get("訂單記錄",None)
    member_data = fb.get("會員資料",None)
    r = dict()
    f = dict()
    m = dict()
    real_rfm = dict()
    lose_balance_value = dict()
    for key,value in date_data.items():
        if key[14:] in r.keys() :
            pass
        elif count_r >= int(len(member_data)/2) :
            r[key[14:]] = 1
            count_r = count_r + 1
        else :
            r[key[14:]] = 2
            count_r = count_r + 1
    for key,value in date_data.items():
        if key[14:] in m.keys():
            current = m[key[14:]]
            m[key[14:]] = current + int(value["總金額"])
        else:
            m[key[14:]] = int(value["總金額"])
    real_m = sorted(m.items(), key=lambda d: d[1] , reverse = True)
    for value in real_m:
        if count_m >= int(len(member_data)/2) :
            m[value[0]] = 1
            count_m = count_m + 1
        else :
            m[value[0]] = 2
            count_m = count_m + 1
    for key,value in date_data.items():
        if key[14:] in f.keys() :
            current = f[key[14:]]
            if int(key[0:14]) > int(check_time) :
                f[key[14:]] = 1 + current
        else :
            if int(key[0:14]) > int(check_time) :
                f[key[14:]] = 1
    real_f = sorted(f.items(), key=lambda d: d[1] , reverse = True)
    for value in real_f:
        if count_f >= int(len(member_data)/2) :
            f[value[0]] = 1
            count_f = count_f + 1
        else :
            f[value[0]] = 2
            count_f = count_f + 1
    for key,value in r.items():
        real_rfm[key] = str(r[key]) + str(f[key]) + str(m[key])
    for i in range(8):
        lose_balance_value[level[i]] = round(float(float(response[i]) - float(lose_balance))/float(lose_balance) , 2)

    real_response = {
        "222" : 0.3 ,
        "221" : 0.2 ,
        "212" : 0.08,
        "211" : 0.055,
        "122" : 0.03,
        "121" : 0.015,
        "112" : 0.009,
        "111" : 0.0015
    }
    return real_rfm , lose_balance_value , delivery_cost , per_profit , lose_balance ,real_response , level
