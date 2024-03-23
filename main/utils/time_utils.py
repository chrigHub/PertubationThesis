import datetime


def validate_time(date_string, format: str = '%m-%d-%Y %H:%M:%S'):
    ret = True
    try:
        time = datetime.datetime.strptime(date_string, format)
        return ret, time
    except ValueError:
        ret = False
        return ret, None


def print_time(time_format: str = "%Y_%m_%d %H:%M") -> (datetime, str):
    time = datetime.datetime.now()
    return time, time.strftime(time_format)
