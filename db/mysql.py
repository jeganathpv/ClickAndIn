import pymysql
connection = pymysql.connect('localhost','newuser','password','click')
cursor=connection.cursor()
# sql = """CREATE TABLE attendance (
#    name  CHAR(20) NOT NULL,
#    date  CHAR(20),
#    attend INT )"""
# cursor.execute(sql)
# data=cursor.fetchall()
# print(data)
name="Faze"
time="12-12-123"
sql="INSERT INTO attendance values('{}','{}',{})".format(name,time,1)
cursor.execute(sql)
connection.close()