import json
import time
from datetime import datetime

start_time = time.time()
date_time = datetime.fromtimestamp(time.time())
readable_date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")

stat_data = {
    "start_time": readable_date_time,
    "time_elapsed": 0,
    'all': {
        'size': 0,
        'average_time_taken': 0,
        'median_time': 0,
        'gatekeeper_accuracy': 0,
        'cost': 0,
    },
    'gpt3': {
        'data': [],
        'size': 0,
        'average_time_taken': 0,
        'median_time': 0,
        'cost': 0,
    },
    'gpt4': {
        'data': [],
        'size': 0,
        'average_time_taken': 0,
        'median_time': 0,
        'cost': 0,
    },
    'image': {
        'data': [],
        'size': 0,
        'average_time_taken': 0,
        'median_time': 0,
    },
    'gatekeeper': {
        'data': [],
        'gatekeeper_accuracy': 0,
    },
}

def write():
    filename = "definer_stats/stats/stats {}.json".format(readable_date_time)
    with open(filename, 'w') as file:
        json.dump(stat_data, file, indent=4)

#############

def get_time_elapsed():
    elapsed = datetime.fromtimestamp(time.time() - start_time)
    return elapsed.strftime("%H:%M:%S")

def track_gpt3_time_and_cost(time_taken, amount_to_increase):
    stat_data['gpt3']['data'].append(time_taken)
    stat_data['gpt3']['size'] += 1
    stat_data['gpt3']['cost'] += amount_to_increase
    stat_data['all']['size'] += 1
    stat_data['all']['cost'] += amount_to_increase

    gpt3times = stat_data['gpt3']['data']
    gpt3times.sort()
    n = len(gpt3times)
    if n % 2 == 0:
        median = (gpt3times[n//2 - 1] + gpt3times[n//2]) / 2
    else:
        median = gpt3times[n//2]

    stat_data['gpt3']['average_time_taken'] = sum(gpt3times) / len(gpt3times)
    stat_data['gpt3']['median_time'] = median
    write()

def track_gpt4_time_and_cost(time_taken, amount_to_increase):
    stat_data['gpt4']['data'].append(time_taken)
    stat_data['gpt4']['size'] += 1
    stat_data['gpt4']['cost'] += amount_to_increase
    stat_data['all']['size'] += 1
    stat_data['all']['cost'] += amount_to_increase

    gpt4times = stat_data['gpt4']['data']
    gpt4times.sort()
    n = len(gpt4times)
    if n % 2 == 0:
        median = (gpt4times[n//2 - 1] + gpt4times[n//2]) / 2
    else:
        median = gpt4times[n//2]

    stat_data['gpt4']['average_time_taken'] = sum(gpt4times) / len(gpt4times)
    stat_data['gpt4']['median_time'] = median
    write()

def track_image_time(time_taken):
    stat_data['image']['data'].append(time_taken)
    stat_data['image']['size'] += 1
    image_times = stat_data['image']['data']
    image_times.sort()
    n = len(image_times)
    if n % 2 == 0:
        median = (image_times[n//2 - 1] + image_times[n//2]) / 2
    else:
        median = image_times[n//2]
    stat_data['image']['average_time_taken'] = sum(image_times) / len(image_times)
    stat_data['image']['median_time'] = median

def gatekeeper_accuracy_average(accuracy):
    if accuracy is None: return
    stat_data['gatekeeper']['data'].append(accuracy)
    data = stat_data['gatekeeper']['data']
    stat_data['gatekeeper']['gatekeeper_accuracy'] = sum(data) / len(data)
    stat_data['all']['gatekeeper_accuracy'] = stat_data['gatekeeper']['gatekeeper_accuracy']
    write()

times = []
def track_time_average(time_taken):
    if time_taken is None: return
    times.append(time_taken)
    times.sort()
    n = len(times)
    if n % 2 == 0:
        median = (times[n//2 - 1] + times[n//2]) / 2
    else:
        median = times[n//2]

    stat_data['all']['average_time_taken'] = sum(times) / len(times)
    stat_data['all']['median_time'] = median
    stat_data['time_elapsed'] = get_time_elapsed()
    write()

