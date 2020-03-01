import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

running_email = "104306093@nccu.edu.tw"

msg = MIMEMultipart('alternative')
msg['Subject'] = "2018-06-11 台北政大系內馬拉松競賽!!!"
msg['From'] = running_email
msg['to'] = "maxwell111023@gmail.com"
me = "maxwell111023@gmail.com"

# Create the body of the message (a plain-text and an HTML version).
html = """\
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
"""


