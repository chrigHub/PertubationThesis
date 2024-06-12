import datetime


def validate_time(date_string, format: str = '%m-%d-%Y %H:%M:%S'):
    """
    Checks whether ot not a given time string fits a certain format.
    :param date_string: The string to be checked.
    :param format: The format on which the timestring is being checked
    :return: Returns tuple of whether parsing was possible + time instance.
        (boolean, instance)
        Instance is None if parsing failed.
    """
    ret = True
    try:
        time = datetime.datetime.strptime(date_string, format)
        return ret, time
    except ValueError:
        ret = False
        return ret, None


def print_time(time_format: str = "%Y_%m_%d %H:%M") -> (datetime, str):
    """
    Returns time instance and string from the time of call
    :param time_format: The string format for the output
    :return: Tuple of time instance and time string on time of function call.
        (instance, string)
    """
    time = datetime.datetime.now()
    return time, time.strftime(time_format)
