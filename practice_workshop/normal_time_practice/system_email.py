#import some specific library that we need
from geopy.distance import geodesic
import pyrebase,json
import csv,json,jieba,re,glob,sys
import datetime
import os, os.path
from shutil import copyfile

# initialize our running app
config_running_app = {
    "apiKey": "AIzaSyApHVzGEc9TYksZZV8dnKSOiIucxiEnQUU",
    "authDomain": "runningmate-7c3f2.firebaseapp.com",
    "databaseURL": "https://runningmate-7c3f2.firebaseio.com",
    "projectId": "runningmate-7c3f2",
    "storageBucket": "runningmate-7c3f2.appspot.com",
    "messagingSenderId": "867235253757"
}
running_app_firebase = pyrebase.initialize_app(config_running_app)
running_app_db = running_app_firebase.database()

sending_email = "system_email.csv"

recent_members = list()
email_to_send = dict()

#將『 key 』設成『 第 X 筆 』
with open(sending_email,"r") as email :
    email_data = csv.DictReader(email)

    members = running_app_db.child("members").get()
    recent_members =  list(members.val().keys())

    # 將原始檔案中的每行資料讀進每個 key 成為 value 值
    for i in range(len(recent_members)) :

        email_to_send = dict()
        email_person = running_app_db.child("email").child(recent_members[i]).get()
        
        if email_person.each() == None :
            current_count = 1
        else :
            current_count = len(email_person.val()) + 1

        for line in email_data :
            if line["收件者"] == "All" or line["收件者"] == recent_members[i] :
                email_to_send["第"+str(current_count)+"封"] = dict()
                email_to_send["第"+str(current_count)+"封"]["標題"] = line["標題"]
                email_to_send["第"+str(current_count)+"封"]["寄件者"] = "Running Mate"
                email_to_send["第"+str(current_count)+"封"]["收件時間"] = datetime.datetime.now().strftime("%Y-%m-%d")
                email_to_send["第"+str(current_count)+"封"]["內容"] = line["內容"]
                email_to_send["第"+str(current_count)+"封"]["狀態"] = 0
            current_count = current_count + 1 

        if email_person.each() == None :
            running_app_db.child("email").child(recent_members[i]).set(email_to_send)
        else :
            running_app_db.child("email").child(recent_members[i]).update(email_to_send)

        email.seek(0)
        next(email)

    # print(email_to_send)

    email_to_send = dict()
    email.close()

# # create directory if not exsist
# if not os.path.exists("email_history"): 
#     os.mkdir("email_history")

# history_count = (len([name for name in os.listdir("/Users/wangboren/python3/practice_workshop/normal_time_practice/email_history") if name.endswith('.' + "csv")]))

# copyfile("system_email.csv", "system_email_history"+ str(history_count+1) +".csv")

# os.remove('/Users/wangboren/python3/practice_workshop/normal_time_practice/system_email.csv')

