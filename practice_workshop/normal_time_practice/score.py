import os,time,keyboard,json,hashlib,ast,numpy,menu
import matplotlib.pyplot as plt

def check_again(name,score):
    reply = input("Did {} get {} ?,is that right ? (Press Y/y or N/n)".format(name,score))
    while True :
        if reply == "Y" or reply == "y":
            return 1
        elif reply == "N" or reply == "n":
            return 0
        else:
            print("Enter a wrong word !")
            reply = input("Did {} get {} ?,is that right ? (Press Y/y or N/n)".format(name,score))

def enter_score(subject,score):
    print("You can reivse student's score here or enter new one !")
    subject_list = dict()
    score_list = dict()
    while True:
        name = input("Student's Name : ")
        if name not in score.keys():
            score[name] = {}
            score[name]["成績"] = {}
            score[name]["資料"] = {}
            if subject not in score[name]["成績"].keys():
                score[name]["成績"][subject] = {}
        else:
            if subject not in score[name]["成績"].keys():
                score[name]["成績"][subject] = {}

        kind = input("What kind of {} test : ".format(subject))
        if not score[name]["成績"][subject].keys():
            score[name]["成績"][subject] = {}
            score[name]["成績"][subject][kind] = {}
        else:
            if kind not in score[name]["成績"][subject].keys():
                score[name]["成績"][subject][kind] = {}

        which = input("第幾次{}？（數字）".format(kind))
        score_student = input("Student {}'s 第{}次{}{} Score : ".format(name,which,subject,kind))
        score[name]["成績"][subject][kind]["第" + str(which) + "次"] = score_student
        print(score)
        with open("score.json","w+",encoding="UTF-8-sig") as f:
            f.write(json.dumps(score,ensure_ascii=False,indent=4))
            f.close()

        go_or_not = input("Continue to enter ? (y/n)")
        if go_or_not == "y" or go_or_not =="Y":
            pass
        elif go_or_not == "n" or go_or_not =="N":
            break
        else:
            print("Enter Again...(Wait for few second)")
            time.sleep(3)

def View_score(subject,score):
    if not score:
        print("There's no data !")
    else:
        for name, _score in score.items():
            print("這是{}的{}歷次成績：".format(name,subject))
            print(_score["成績"])
            print(_score.keys())
    input("Press any key to leave ...")

def Delete_score(subject,score):
    name = input("Who's score do you want to delete ? ")
    if name in score:
        score.pop(name)
    else:
        print("There's no data !")
    input("Press any key to leave ...")
