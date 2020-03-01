# 匯入urllib 庫
import urllib.request
import time
import urllib.parse
import ssl
import requests
from bs4 import BeautifulSoup as bs
import re


# url 地址
URL = "http://kns.cnki.net/kns/brief/default_result.aspx"
keyword = '上海新漫传感技术研究发展有限公司'

cookies = {'ASP.NET_SessionId':'qdightqtk5p4lbd2pnsenypj',
           'Ecp_ClientId':'8190713151002814189',
           'Ecp_LoginStuts':'%7B%22IsAutoLogin%22%3Afalse%2C%22UserName%22%3A%22nccu%22%2C%22ShowName%22%3A%22%25E6%2594%25BF%25E6%25B2%25BB%25E5%25A4%25A7%25E5%25AD%25A6%22%2C%22UserType%22%3A%22bk%22%2C%22r%22%3A%22Wuejhb%22%7D	',
           'Ecp_notFirstLogin':'Wuejhb',
           'Ecp_session':'1',
           'LID':'WEEvREdxOWJmbC9oM1NjYkZCbDdrdXJJcEhaMXN6aFFEQmZlNHJhVW9GbWc',
           'SID_crrs':'125133',
           'SID_klogin':'125144',
           'SID_kns':'123120',
           'SID_krsnew':'125132'
        #    'c_m_LinID':'LinID=WEEvREdxOWJmbC9oM1NjYkZCbDdrdXJJcEhaMXN6aFFEQmZlNHJhVW9GbWc=$R1yZ0H6jyaa0en3RxVUd8df-oHi7XMMDo7mtKT6mSmEvTuk11l2gFA!!&ot=07/13/2019 16:59:42',
        #    'c_m_expire':'2019-07-13 16:59:42'
           #'cnkiUserKey':'c744db90-9610-832c-587f-07c698b8c52b'
           }

#brief
data6 = {'pagename': 'ASP.brief_default_result_aspx',
        'isinEn':'0',
        'dbPrefix': 'SCOD',
        'dbCatalog': '中国学术文献网络出版总库',
        'ConfigFile': 'SCDBINDEX.xml',
        'research': 'off',
        't': int(time.time()),
        'keyValue': keyword,
        'S': '1'
        }
query_string6 = urllib.parse.urlencode(data6)
url6 = 'https://kns.cnki.net/kns/brief/brief.aspx' + '?' + query_string6
print(url6)
result6 = requests.get(url6, cookies=cookies, headers=dict(referer=URL))
soup = bs(result6.text, 'html.parser')

print(soup.prettify())