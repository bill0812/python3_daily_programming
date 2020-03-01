import os,time,keyboard,json,hashlib,ast,numpy,menu,compute
import matplotlib.pyplot as plt
import score as ss
def Subject_system(score):
    os.system("clear")
    while True:
        menu.select_subject()
        choice = input("Enter your choice:\n=>")
        if choice == "C" or choice == "c" :
            decide_action("中文",score)
        elif choice == "E" or choice == "e":
            decide_action("英文",score)
        elif choice == "M" or choice == "m":
            decide_action("數學",score)
        elif choice == "So" or choice == "sc" or choice == "SO":
            decide_action("社會",score)
        elif choice == "Sc" or choice == "so" or choice == "SC":
            decide_action("自然",score)
        elif choice == "Q" or choice == "q":
            break
        else:
            print("Enter Again...(Wait for few second)")
            time.sleep(3)

def Score_system(score):
    os.system("clear")
    while True:
        menu.select_compute()
        choice = input("Enter your choice:\n=>")
        if choice == "student":
            Compute_system("student",score)
        elif choice == "subject":
            Compute_system("subject",score)
        elif choice == "Q" or choice == "q":
            break
        else:
            print("Enter Again...(Wait for few second)")
            time.sleep(3)

def Compute_system(way,score):
    os.system("clear")
    while True:
        menu.Compute()
        choice = input("Enter your choice:\n=>")
        if choice == "S" or choice == "s":
            compute.standard_score(way,score)
        elif choice == "M" or choice == "m":
            compute.mean_score(way,score)
        elif choice == "MM" or choice == "mm":
            compute.median_score(way,score)
        elif choice == "K" or choice == "k":
            compute.clustering_score(way,score)
        elif choice == "P" or choice == "p":
            compute.plotting_score(way,score)
        elif choice == "Q" or choice == "q":
            break
        else:
            print("Enter Again...(Wait for few second)")
            time.sleep(3)

def decide_action(subject,score):
    os.system("clear")
    while True:
        menu.Score(subject)
        choice = input("Enter your choice:\n=>")
        if choice == "E" or choice == "e":
            ss.enter_score(subject,score)
        elif choice == "V" or choice == "v":
            ss.View_score(subject,score)
        elif choice == "D" or choice == "d":
            ss.Delete_score(subject,score)
        elif choice == "Q" or choice == "q":
            break
        else:
            print("Enter Again...(Wait for few second)")
            time.sleep(3)
