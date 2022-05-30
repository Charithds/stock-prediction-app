import pandas as pd

def addTrainingJob(company, conn):
    conn.execute('INSERT INTO training_job(bank,tag,training_file,model_file) VALUES (?,?,?,?)',
                 (company['bank'], company['tag'], company['trainingFile'], company['modelFile']))
    conn.commit()


def getTrainingJobs(conn):
    c = conn.cursor()
    c.execute('SELECT * FROM training_job;')
    return pd.DataFrame(c.fetchall())


def createTables(conn):
    conn.execute(
        'CREATE TABLE IF NOT EXISTS training_job (id INTEGER PRIMARY KEY AUTOINCREMENT, bank CHAR(100), tag CHAR(256), training_file TEXT, model_file TEXT, CONSTRAINT uniq_tag UNIQUE (bank, tag));')
    conn.commit()
