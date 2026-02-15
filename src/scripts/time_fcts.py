import logging

def print_time(start,end):
    time = end - start

    days = time // (86400)
    hours = (time % 86400)// 3600
    mins = (time % 3600) // 60
    secs = time % 60
    logging.info(f"{days} day(s), {hours} hour(s), {mins} min(s) and {secs} sec(s).\n")