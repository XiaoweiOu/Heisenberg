import time

def time_it(f):
    
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = f(*args, **kwargs)
        t_end = time.time()
        t = t_end - t_start
        print("%s took %.2f sec" % (f.__name__, t))
        return result

    return wrapper