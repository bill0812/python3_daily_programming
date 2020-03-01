import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

running_email = "104306093@nccu.edu.tw"

msg = MIMEMultipart('alternative')
msg['Subject'] = "2018-06-11 資管 Running Mate 首度公開!!!"
msg['From'] = running_email
msg['to'] = "maxwell111023@gmail.com"
me = "maxwell111023@gmail.com"

# Create the body of the message (a plain-text and an HTML version).
html = """\
<html>

  <head>
    <link href="https://fonts.googleapis.com/css?family=Noto+Sans" rel="stylesheet">
  </head>
  <style>
    .body{{
        width : 750px;
        font-family: 'Noto Sans', sans-serif !important;
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
  <body style = "font-family: 'Noto Sans', sans-serif !important;width : 750px;">
    <h2 class = "title" style = "color : #E9A11A;">資管「 Running Mate 」首度公開啦 ！！在 6 月 11 號 ， Running Mate 歡迎大家來共襄盛舉哦～～</h2>
    <p class = "content" style = "font-size: 18px;color : black;font-weight: 400;">Hi {name} 您好, 最近忙著期末的你，事否有持續在運動呢？</p>
    <p class = "content" style = "font-size: 18px;color : black;font-weight: 400;">在 6 月 11 號 活動當天，Running Mate 將會在
    「 玉山國際會議廳 」發表最新的 「 跑步交友 」軟體，首度亮相！</p>
    <br>
    <br>
    <p class = "content" style = "font-size: 18px;color : black;font-weight: 400;display: flex;line-height: 30px;">
        <img src = "https://imgur.com/rAZwsfY.png" style = "width: 30px;height: 30px;margin-right: 15px;"/>Running Mate 結合了：    
    </p>
    <ul class = "content" style = "font-size: 18px;color : black;font-weight: 400;">
        <li style = "list-style: none;display: flex;line-height: 30px;margin-bottom: 10px;"><img src = "https://imgur.com/9cXny8v.png" style = "width: 30px;height: 30px;margin-right: 15px;"/>遊戲</li>
        <li style = "list-style: none;display: flex;line-height: 30px;margin-bottom: 10px;"><img src = "https://imgur.com/CvAcZNJ.png" style = "width: 30px;height: 30px;margin-right: 15px;"/>交友</li>
        <li style = "list-style: none;display: flex;line-height: 31px;margin-bottom: 10px;"><img src = "https://imgur.com/tRzkSxx.png" style = "width: 30px;height: 30px;margin-right: 15px;"/>運動</li>
    </ul>

    <br>
    <p class = "content" style = "font-size: 18px;color : black;font-weight: 400;">「 Running Mate 」相信能一定成為時下瘋狂的新穎 APP!!</p>
    
    <br>
    <p class = "content" style = "font-size: 18px;color : black;font-weight: 400;">想當然的，也會有許多驚喜等著各位呢！Running Mate 屆時將會與您共享歡樂！</p>

    <p class = "star-content" style = "font-size: 18px;color : black;text-align : center;">各位朋友們，我們不見不散哦！！</p>
    <h2 class = "subtitle last" style = "color : #E9A11A;text-align : right;">Running Mate,  sincerely</h2>
    <img style = "width: 100%;" src = "https://imgur.com/uxE4QDv.png"/>
  </body>
  
</html>
"""


html = html.format(name="親愛的同學")
# Record the MIME types of both parts - text/plain and text/html.
part1 = MIMEText(html, 'html')

# Attach parts into message container.
# According to RFC 2046, the last part of a multipart message, in this case
# the HTML message, is best and preferred.
msg.attach(part1)
# Send the message via local SMTP server.
s = smtplib.SMTP("localhost")
# sendmail function takes 3 arguments: sender's address, recipient's address
# and message to send - here it is sent as one string.
s.sendmail(running_email, me, msg.as_string())
s.quit()