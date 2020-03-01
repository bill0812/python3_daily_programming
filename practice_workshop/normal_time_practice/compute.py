import os,time,keyboard,json,hashlib,ast,numpy,menu,score,compute
import matplotlib.pyplot as plt

def standard_score(way,score):
    student_score = list()
    for key, value in score.items():
        student_score.append(int(value))

    std = numpy.std(numpy.array(student_score))
    print("The student's standard score is : {}".format(std))
    input("Press any key to leave ...")

def mean_score(way,score):
    student_score = list()
    for key, value in score.items():
        student_score.append(int(value))

    std = numpy.mean(numpy.array(student_score))
    print("The student's mean score is : {}".format(std))
    input("Press any key to leave ...")

def median_score(way,score):
    student_score = list()
    for key, value in score.items():
        student_score.append(int(value))

    std = numpy.median(numpy.array(student_score))
    print("The student's median score is : {}".format(std))
    input("Press any key to leave ...")

def clustering_score(way,score):
    student_score = list()
    for key, value in score.items():
        student_score.append(int(value))

    std = numpy.std(numpy.array(student_score))
    print("The student's standard score is : {}".format(std))
    input("Press any key to leave ...")

def plotting_score(way,score):
    student_score = list()
    for key, value in score.items():
        student_score.append(int(value))

    std = numpy.std(numpy.array(student_score))
    print("The student's standard score is : {}".format(std))
    input("Press any key to leave ...")
