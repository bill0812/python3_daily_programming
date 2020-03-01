import os,time,keyboard,json,hashlib,ast,numpy,menu
import system_choose as sc
import score as ds
import matplotlib.pyplot as plt
if __name__ == "__main__":
    score = dict()
    with open("score.json","r",encoding = "UTF-8-sig") as f:
        some_string = f.read()
        if not some_string:
            score = {}
        else:
            score = json.loads(some_string)

    while(True):
        menu.menu()
        choice = input("Enter your choice:\n=>")
        if choice == "E" or choice == "e" :
            sc.Subject_system(score)
        elif choice == "C" or choice == "c":
            sc.Score_system(score)
        elif choice == "Q" or choice == "q":
            break
        else:
            print("Enter Again...(Wait for few second)")
            time.sleep(3)
