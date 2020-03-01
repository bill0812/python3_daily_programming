from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import timedelta
from datetime import datetime
from datetime import date
import salesapp.login as login_status
import salesapp.product as product
import salesapp.register as register_status
import salesapp.reply as reply_data
import salesapp.customer as customer_data
import salesapp.delivery as delivery_data
import salesapp.cart as cart_data
import salesapp.order as order_data
import salesapp.personal as personal_data
import salesapp.admin_page as admin_page
import time,re
import json
# Create your views here.
account,password,name  = "","","訪客"
customer_status,normal_status = list() , list()
phone = ""
address = ""
customer = dict()
normal = dict()
user = 0
def login(request):
    global user
    return render(request,"login.html",{"user":user,"name":name})

def register_address(request):
    if request.method == "GET":
        if request.is_ajax():
            county =  request.GET["county"]
            district = register_status.fetch_district(county)
            district_array = district.split(",")
            return JsonResponse(district_array, safe=False)
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def register_member(request):
    global account
    global password
    global name
    global user
    if request.method == "POST":
        if request.is_ajax():
            name = request.POST['name']
            account = request.POST['email']
            password = request.POST['password']
            sex = request.POST['sex']
            phone = request.POST['phone']
            address = request.POST['address']
            check_register,status,description = register_status.update_to_firebase(name,account,password,sex,phone,address)
            if check_register == 1 :
                account = status
                name = description
                user = 1
                return HttpResponse("success")
            else:
                account,password,name  = "","","訪客"
                user = 0
                return HttpResponse("fail")
    else:
        user = 0
        return HttpResponse("HTTP Went Wrong !")

def register(request):
    global user
    address = register_status.fetch_address()
    return render(request,"register.html",{"user":user,"address":address})

def vertify_account(request):
    global account
    account = request.GET['account']
    check_account = login_status.check_account_exsist(account)
    if check_account == 1 :
        return HttpResponse("success")
    else:
        account,password,name  = "","","訪客"
        return HttpResponse("no_account")

def vertify(request):
    global account
    global password
    global name
    global user
    if request.method == "POST":
        if request.is_ajax():
            account = request.POST['account']
            password = request.POST['password']
            check_login,key,name_status = login_status.check_member_exsist(account,password)
            if check_login == 1 :
                account = key
                name = name_status
                user = 1
                return HttpResponse("success")
            elif name_status == "帳號不存在！":
                account,password,name  = "","","訪客"
                user = 0
                return HttpResponse("no_member")
            elif name_status == "錯誤密碼":
                account,password,name  = "","","訪客"
                user = 0
                return HttpResponse("wrong")
            else:
                account,password,name  = "","","訪客"
                user = 0
                return HttpResponse("fail")
    else:
        user = 0
        return HttpResponse("HTTP Went Wrong !")

def main_page(request):
    global user
    global name
    product_data = product.fetch_product()
    return render(request,"main.html",{"user":user,"name":name,"product_data":product_data})

def reply(request):
    global user
    global name
    return render(request,"reply.html",{"user":user,"name":name})

def reply_status(request):
    global account
    words = request.GET['words']
    reply_data.upload_words(words,account)
    return HttpResponse("success")

def logout(request):
    global user
    global name
    user = 0
    name = "訪客"
    return redirect("/main/")

def customer_page(request):
    global user
    global name
    status = customer_data.customer_show()
    if status == 0 :
        product = []
    else :
        product = status
    return render(request,"customer.html",{"user":user,"name":name,"product":product})

def customer_enter(request):
    customer_count = dict()
    if request.method == "GET":
        if request.is_ajax():
            for key,value in request.GET.items():
                customer_count[key] = value
            confirm = customer_data.upload_to_cart(name,account,customer_count)
            if confirm == 1:
                return HttpResponse("success")
            else:
                return HttpResponse("fail")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def delivery(request):
    global user
    global name
    global account
    global customer_status
    global normal_status
    global customer
    global normal
    customer = delivery_data.fetch_customer_product(account,customer_status)
    normal = delivery_data.fetch_normal_product(account,normal_status)
    status = customer_data.customer_show()
    if status == 0:
        product = []
    else:
        product = status
    phone,address = delivery_data.fetch_phone_address(account)
    return render(request,"delivery.html",{"user":user,"name":name,"phone":phone,"address":address,"product":product,"customer":customer,"normal":normal})

def cart(request):
    global user
    global name
    global account
    customer,normal_product = cart_data.fetch_cart(account)
    status = customer_data.customer_show()
    if status == 0:
        product = []
    else:
        product = status
    return render(request,"cart.html",{"user":user,"name":name,"customer":customer,"product":product,"normal_product":normal_product})

def cart_upload(request):
    global customer_status
    global normal_status
    global account
    if request.method == "GET":
        if request.is_ajax():
            customer_status = request.GET.getlist("customer[]")
            normal_status = request.GET.getlist("normal[]")
            return HttpResponse("success")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def make_order(request):
    global name
    global customer_status
    global normal_status
    global phone
    global address
    global order_name
    global order_time
    global account
    if request.method == "GET":
        if request.is_ajax():
            order_name = request.GET["name"]
            phone = request.GET["phone"]
            address = request.GET["address"]
            order_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            order_data.make_order(account,order_time,name,order_name,phone,address,customer_status,normal_status)
            return HttpResponse("success")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def upload_normal(request):
    global account
    if request.method == "GET":
        if request.is_ajax():
            name = request.GET["name"]
            count = request.GET["count"]
            cart_data.upload_normal(name,count,account)
            return HttpResponse("success")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def order(request):
    global name
    global customer_status
    global normal_status
    global phone
    global address
    global order_name
    global order_time
    global account
    global customer
    global normal
    status = customer_data.customer_show()
    if status == 0:
        product = []
    else:
        product = status
    return render(request,"order.html",{"user":user,"name":name,"number":order_time+name,"time":order_time,"name":order_name,"product":product,"phone":phone,"address":address,"customer":customer,"normal":normal})

def personal(request):
    global account
    global user
    global name
    personal,order = personal_data.fetch_person(account)
    return render(request,"personal.html",{"user":user,"name":name,"personal":personal,"order":order})

def edit_data(request):
    if request.method == "GET":
        if request.is_ajax():
            name = request.GET["name"]
            kind = request.GET["kind"]
            personal_data.edit_data(account,name,kind)
            return HttpResponse("success")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def admin(request):
    return render(request,"admin.html",{"user":user,"name":"管理者"})

def admin_page_order(request):
    order = admin_page.fetch_order
    return render(request,"admin_page_order.html",{"order":order})

def admin_page_structure(request):
    today_time = datetime.today().weekday()
    status,left,median = admin_page.inventory_situation(today_time)
    product_inventory = admin_page.fetch_inventory()
    product_data = product.fetch_product()
    return render(request,"admin_page_structure.html",{"product":product_inventory,"product_data":product_data,"status":status,"left":left,"median":median})

def sned_inventory(request):
    if request.method == "GET":
        if request.is_ajax():
            quantity = request.GET["quantity"]
            day = request.GET["day"]
            name = request.GET["name"]
            compose_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            count_time =  datetime.now() + timedelta(days = int(day[:-1]))
            ending_time = count_time.strftime("%Y%m%d%H%M%S")
            check = admin_page.send_inventory_order(name,quantity,compose_time,ending_time)
            if check == 1:
                return HttpResponse("success")
            else:
                return HttpResponse("fail")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def product_structure(request):
    if request.method == "GET":
        if request.is_ajax():
            id = request.GET["id"]
            product_inventory = admin_page.fetch_product(id)
            return JsonResponse(product_inventory, safe=False)
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def compose_product(request):
    if request.method == "GET":
        if request.is_ajax():
            title = request.GET["title"]
            quantity = request.GET["quantity"]
            make_time = request.GET["time"]
            compose_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            count_time =  datetime.now() + timedelta(days = float(make_time[:-1]))
            ending_time = count_time.strftime("%Y%m%d%H%M%S")
            check = admin_page.compose_product(title,quantity,compose_time,ending_time)
            if check == 1:
                return HttpResponse("success")
            else:
                return HttpResponse("fail")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def adminprocess(request):
    pro_data,pro_status,veg_data,veg_status = admin_page.fetch_inbound()
    print(veg_status)
    return render(request,"process_admin.html",{"user":user,"name":"管理者","pro_data":pro_data,"pro_status":pro_status,"veg_data":veg_data,"veg_status":veg_status})

def get_product_order(request):
    if request.method == "GET":
        if request.is_ajax():
            title = request.GET["order"]
            check = admin_page.get_product_order(title)
            if check == 1:
                return HttpResponse("success")
            else:
                return HttpResponse("fail")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def get_veg_order(request):
    if request.method == "GET":
        if request.is_ajax():
            title = request.GET["order"]
            check = admin_page.get_veg_order(title)
            if check == 1:
                return HttpResponse("success")
            else:
                return HttpResponse("fail")
        else:
            return HttpResponse("fail")
    else:
        return HttpResponse("HTTP Went Wrong !")

def show_rfm(request):
    real_rfm , lose_balance_value , delivery_cost , per_profit , lose_balance ,real_response ,level = admin_page.rfm()
    return render(request,"adminRFM.html",{"user":user,"name":"管理者","real_rfm":real_rfm,"lose_balance_value":lose_balance_value,"delivery_cost":delivery_cost,"per_profit":per_profit,"lose_balance":lose_balance ,"real_response":real_response,"level":level})
