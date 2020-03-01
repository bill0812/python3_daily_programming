from django.shortcuts import render,redirect
from django.http import HttpResponse
account = str()
# Create your views here.
def home_page(request):
    return render(request,"home_page.html")

def register(request):
    return render(request,"register.html")

def main_page(request):
    name = account



    return render(request,"main_page.html",locals())

def vertify(request):
    global account
    if request.method == "POST":
        account = request.POST["account"]
        password = request.POST["password"]
        if account == "123" and password == "123":
            return redirect("/main_page/")
        else:
            return HttpResponse("hey")
    else:
        return HttpResponse("hey")
