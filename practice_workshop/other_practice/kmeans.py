import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint 
import random, uuid, types, math

# define some data set
df = pd.read_csv('summer-travel-gps-full.csv')
coords = df.as_matrix(columns=['lat', 'lon'])
main_pool_test = {
    "bill" : {
        "x" : 51.4812916,
        "y" : -0.4510112
    },
    "henry" : {
        "x" : 51.474005,
        "y" : -0.4509991
    },
    "deeana" : {
        "x" : 51.4781991,
        "y" : -0.446081
    },
    "tony" : {
        "x" : 51.4801463,
        "y" : -0.4411027
    },
    "cory" : {
        "x" : 38.7071923,
        "y" : -9.1368245
    }
}
left_pool_test = dict()
main_pool_list = list()
left_pool = list()
room = dict()
players = dict()
kms_per_radian = 6371.0088
epsilon = 2 / kms_per_radian

# retrieve data (x,y) from each user
def retrieve_user_data(main_pool_test) :

    for key, value in main_pool_test.items() :
        main_pool_list.append([value["x"],value["y"]])

    circle_clustering(main_pool_list)

# clustering main 
def circle_clustering(main_pool_list):
    # define clustering parameters
    
    global kms_per_radian
    global epsilon

    db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(np.array(main_pool_list)))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([np.array(main_pool_list)[cluster_labels == n] for n in range(num_clusters)])

    seperate_room(cluster_labels,num_clusters,clusters)

# circle center
def center_circle(each_cluster) :
    random_angle = random.uniform(0,360) #角度 0~360
    c1_center = each_cluster.mean(axis=0)
    L1 = random.uniform(0,1.5) 
    c2x = c1_center[0] + L1 * math.cos(math.radians(random_angle))
    c2y = c1_center[1] + L1 * math.sin(math.radians(random_angle))
    L2 = random.uniform(0,0.75)
    c3x = c2x + L2 * math.cos(random_angle)
    c3y = c2y + L2 * math.sin(random_angle)
    L3 = random.uniform(0,0.65)
    c4x = c3x + L3 * math.cos(random_angle)
    c4y = c3y + L3 * math.sin(random_angle)
    return c1_center , np.array([c2x,c2y]) , np.array([c3x,c3y]) , np.array([c4x,c4y]) 

# find and return members
def find_members(room_data) :
    
    global main_pool_test
    players_key_each_room = list()

    for i in range(len(room_data)) :
        for key,value in main_pool_test.items() :
            if int(room_data[i][0]) == int(value["x"]) and int(room_data[i][1]) == int(value["y"]) and key not in players_key_each_room:
                players_key_each_room.append(key)
                break
            else :
                pass

    # print(players_key_each_room)
    return players_key_each_room

# insert for each players
def insert_players(players_key_each_room,c1_center,c2_center,c3_center,c4_center):
    
    global players
    global main_pool_test

    for i in range(len(players_key_each_room)) :
        players[players_key_each_room[i]] = dict()
        players[players_key_each_room[i]]["第一圈圓心"] = c1_center
        players[players_key_each_room[i]]["第二圈圓心"] = c2_center
        players[players_key_each_room[i]]["第三圈圓心"] = c3_center
        players[players_key_each_room[i]]["第四圈圓心"] = c4_center
        players[players_key_each_room[i]]["跑友"] = list()
        for x in range(len(players_key_each_room)) :
            if players_key_each_room[i] == players_key_each_room[x] :
                pass
            else :
                players[players_key_each_room[i]]["跑友"].append(players_key_each_room[x])

    for y in range(len(players_key_each_room)) :
        print(players_key_each_room[y])
        del main_pool_test[players_key_each_room[y]]

def insert_dict_data(room_data) :
    
    global room

    room_id = str(uuid.uuid1())
    room[room_id] = dict()
    c1_center , c2_center , c3_center ,c4_center = center_circle(room_data)
    room[room_id]["經緯度"] = room_data
    room[room_id]["第一圈圓心"] = c1_center
    room[room_id]["第二圈圓心"] = c2_center
    room[room_id]["第三圈圓心"] = c3_center
    room[room_id]["第四圈圓心"] = c4_center
    players_key_each_room = find_members(room_data)
    #player_name = find_names(players_key_each_room)
    # print(players_key_each_room)
    insert_players(players_key_each_room,c1_center,c2_center,c3_center,c4_center)

# cluster_data is a list
def room_create(cluster_data) :
    if isinstance(cluster_data, list):
        for i in range(len(cluster_data)):
            insert_dict_data(cluster_data[i])
    else :
        insert_dict_data(cluster_data)

# seperate data to each room
def seperate_room(cluster_labels,num_clusters,clusters) :
    
    global left_pool

    left_pool.append(list(np.array(main_pool_list)[cluster_labels == -1]))

    for i in range(num_clusters) : 

        if clusters[i].shape[0] == 0 :
            pass
        elif clusters[i].shape[0] > 6 :
            for x in range(6,3,-1):
                room_len_least = (np.array_split(clusters[i], clusters[i].shape[0]/x, 0)[-1]).shape[0]
                if room_len_least >= 3:
                    room_create(np.array_split(clusters[i], clusters[i].shape[0]/x, 0))
                    break
                else :
                    pass
        else:
            room_create((clusters[i]))

# retrieve_user_data(main_pool_test)

# print(room)
# print(left_pool)
# print(players)
# print(main_pool_test)