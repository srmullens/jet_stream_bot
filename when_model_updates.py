
# Need a function that checks to see if the file exists.
#
# Open the list of file times. [YYYYMMDDHHMM,HHMM]
# If YYYYMMDDHHMM model has not been recorded yet, look for the file.
#   If the file exists, record the HHMM time.
# If the model time has been recorded, pass.

from datetime import datetime as dt, timedelta
import csv


def look_for_file(time):
    print(f'In look_for_file: {time:%Y%m%d%H00}')
    try:
        ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg_ana/'
                         f'GFS_Global_0p25deg_ana_{time:%Y%m%d}_{time:%H}00.grib2')
        if ds:
            found_time = dt.utcnow()
            print(f'File found! {found_time:%Y%m%d%H00}')
        return found_time
    except:
        print(f'File not available. Done.')
        return False



# Get current time.
today = dt.utcnow()

if today.hour<6: hour = 0
elif today.hour<12: hour = 6
elif today.hour<18: hour = 12
else: hour = 18

date = dt(today.year, today.month, today.day, hour, 0)
date_6hr_ago = date - timedelta(hours=6)
date_12hr_ago = date - timedelta(hours=12)

date_recorded = f'{date:%Y%m%d%H00}'
date_recorded_6hr_ago = f'{date_6hr_ago:%Y%m%d%H00}'
date_recorded_12hr_ago = f'{date_12hr_ago:%Y%m%d%H00}'


# Open log of results.
rows=[]
print('--> Get model_available.csv')
with open('./model_available.csv') as ma_csv:
    rows_dict = csv.DictReader(ma_csv,delimiter=',')
    for row in rows_dict:
        rows.append(row)

last = rows[-1]
print(f'--> Got file.\n--> Last: {last}')

# Search for the file.
if last['run'] == date_recorded:
    print('--> Already found most recent run. Done')
    found = False
elif last['run'] == date_recorded_12hr_ago:
    print(f'--> Looking for run from 12 hours ago: {date_recorded_12hr_ago}')
    model_run = date_12hr_ago
    found = look_for_file(date_12hr_ago)
else:
    print(f'--> Looking for most recent run: {date_recorded_6hr_ago}')
    model_run = date_12hr_ago
    found = look_for_file(date_6hr_ago)


# Log the results, if any.
if isinstance(found, dt):
    print(f'--> Found the most recent run! -> {found:%Y%m%d%H00}\n--> Update model_available.csv')
    with open('./model_available.csv') as ma_csv:
        columns = ['run','available']
        ma = csv.DictWriter(ma_csv,fieldnames=columns)

        ma.writeheader()
        for row in rows:
            ma.writerow(row)
        ma.writerow({'run':f'{model_run:%Y%m%d%H00}','available':f'{found:%Y%m%d%H00}'})
    print('File updated. Done.')

