import os,time,keyboard,json,hashlib,ast,numpy,menu,score
import matplotlib.pyplot as plt

def menu():
    os.system("clear")
    print("This is a system for dealing with students' score :\n")
    print("Press E for entering score\n")
    print("Press C for computing score\n")
    print("Press S for checking students' info\n")
    print("Press Q for leaving the system\n")

def Compute():
    os.system("clear")
    print("This is a function for computing students' score :\n")
    print("Press S for computing the standard of score\n")
    print("Press M for computing the mean of score\n")
    print("Press MM for computing the medium of score\n")
    print("Press K for clustering score\n")
    print("Press P for plotting scatter diagram\n")
    print("Press Q for backing to main system\n")

def select_subject():
    os.system("clear")
    print("This is a place for choosing the subject or different kind's data :\n")
    print("Press C for Chinese\n")
    print("Press E for English\n")
    print("Press M for Math\n")
    print("Press So for Social\n")
    print("Press Sc for Science\n")
    print("Press Q for leaving the system\n")

def select_compute():
    os.system("clear")
    print("Choose a way to compute score :\n")
    print("Press stuednt for student\n")
    print("Press subject for subject\n")
    print("Press Q for leaving the system\n")

def Score(subject):
    os.system("clear")
    print("This is a place for dealing with students' {} score :\n".format(subject))
    print("Press E for adding or revising score\n")
    print("Press V for viewing score\n")
    print("Press D for deleting students' info\n")
    print("Press Q for leaving the system\n")
