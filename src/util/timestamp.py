from datetime import datetime

def time_stamp():
    now = datetime.now()
    time_stamp = now.strftime("%m/%d/%Y, %H:%M:%S").replace("/","-").replace(", ","-").replace(":","-")
    return time_stamp[:-3]

