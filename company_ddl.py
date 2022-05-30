'''
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data'''
import pandas as pd

def addCompany(company, conn):
    conn.execute('INSERT INTO company(code,name) VALUES (?,?)',
                 (company['code'], company['name']))
    conn.commit()


def editCompany(companyId, companyDetails):
    return 2


def getCompanies(conn):
    c = conn.cursor()
    c.execute('SELECT * FROM company;')
    return pd.DataFrame(c.fetchall())


def createTables(conn):
    conn.execute(
        'CREATE TABLE IF NOT EXISTS company (id INTEGER PRIMARY KEY AUTOINCREMENT, code CHAR(100), name CHAR(256));')
    conn.commit()
