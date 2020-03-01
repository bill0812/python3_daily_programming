# import re

# test = '["https://firebasestorage.googleapis.com/v0/b/running-mate-7bb1b.appspot.com/o/airdrops/396e3ba8-6a05-11e8-806a-8c859021ac95/airdrops_1?alt=media"]'

# airdrops_URL_convert = re.findall(r"[\w']+", test)

# test_1 = '"https://firebasestorage.googleapis.com/v0/b/running-mate-7bb1b.appspot.com/o/airdrops/084454c2-6a07-11e8-9a45-8c859021ac95/airdrops_1?alt=media", "https://firebasestorage.googleapis.com/v0/b/running-mate-7bb1b.appspot.com/o/airdrops/084454c2-6a07-11e8-9a45-8c859021ac95/airdrops_1?alt=media"'

# print((test_1))
# print()
# print(re.split(', | ',test_1)[1])

# string = "https://firebasestorage.googleapis.com/v0/b/running-mate-7bb1b.appspot.com/o/airdrops/f0c83e90-6a0f-11e8-8aa3-8c859021ac95/airdrops_1?alt=media&token=8b057d51-5f43-42fb-b029-e83a0c8e2d56"

# front = str(string[:85])

# back = str(string[85:]).replace("/", "%2F")
# print(back)
# print(front)
# https://firebasestorage.googleapis.com/v0/b/running-mate-7bb1b.appspot.com/o/airdrops%2Ff0c83e90-6a0f-11e8-8aa3-8c859021ac95%2Fairdrops_1?alt=media&token=8b057d51-5f43-42fb-b029-e83a0c8e2d56

# import numpy as np

# my_list = np.array([[1,2],[0,2],[2,1],[1,1],[2,2],[2,0],[0,1],[1,0],[0,0]])
# print(my_list.shape[1])
# my_list = sorted(my_list , key=lambda k: [k[1], k[0]])
# print(type(my_list[0]))

# Create the body of the message (a plain-text and an HTML version).
html = '''
<html>

  <head></head>
  <style>
    .body{{
        width : 750px;
    }}
    .title{{
        color : #E9A11A;
    }}
    .subtitle{{
        color : #EA7484;
    }}
    .content{{
        font-size: 18px;
        color : black;
        font-weight: 400;
    }}
    .star-content{{
        font-size: 18px;
        color : black;
        text-align : center;
    }}
    .last{{
        text-align : right;
    }}
  </style>
  <body style = "width : 750px;">
    <h2 class = "title" style = "color : #E9A11A;">資管馬拉松首度舉辦！！在 6 月 11 號 ， Running Mate 歡迎大家來共襄盛舉哦～～</h>
    <p class = "content" style = "font-size: 18px;color : black;font-weight: 400;">Hi {name} , 最近還好嗎 !?</p>
    <p class = "content" style = "font-size: 18px;color : black;font-weight: 400;">親愛的{name}跑友，相信您對於跑步開始產生極大樂趣了吧！當天的馬拉松活動不只要為了成績而戰，中途也設置了許多關卡、空頭等著大家哦！
    想當然的，也會有許多驚喜等著各位呢！Running Mate 屆時將會與您共享歡樂！</p>
    <p class = "star-content" style = "font-size: 18px;color : black;text-align : center;">各位跑友，來賺積分跟禮物吧！！</p>
    <h2 class = "subtitle last" style = "color : #E9A11A;text-align : right;">Running Mate,  sincerely</p>
  </body>
  
</html>
'''
name = ["Bill" , "henry"]
for i in name :
    html1 = html
    html1 = html1.format(name=i)
    
    print(html1)