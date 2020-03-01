#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:16:35 2017

@author: wangboren
"""

import mysql.connector

# 打开数据库连接
db = mysql.connector.connect(host='127.0.0.1',database='test',user='root',password='maxwell13023')

# 使用cursor()方法获取操作游标 
cursor = db.cursor()

# 使用execute方法执行SQL语句
cursor.execute("SELECT*FROM login_database ;")

# 使用 fetchone() 方法获取一条数据库。
data = cursor.fetchone()

print("{}".format(data[0]))

# 关闭数据库连接
db.close()