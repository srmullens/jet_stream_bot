
# Need a function that checks to see if the file exists.
#
# Open the list of file times. [YYYYMMDDHHMM,HHMM]
# If YYYYMMDDHHMM model has not been recorded yet, look for the file.
#   If the file exists, record the HHMM time.
# If the model time has been recorded, pass.

from datetime import datetime as dt, timedelta
import csv
import xarray as xr

def look_for_file(time):
    print(f'  --> In look_for_file: {time}')
    try:
        ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg_ana/'
                         f'GFS_Global_0p25deg_ana_{time:%Y%m%d}_{time:%H}00.grib2')
        if ds:
            found_time = dt.utcnow()
            print(f'  -->File found! {found_time:%Y%m%d%H00}')
        return found_time
    except Exception as e:
        print(f'  --> File not available. Done.')
        print(f'  --> Excpetion:\n{e.args}')
        return False



# Get current time.
today = dt.utcnow()
print(f'--> Current time: {today:%Y%m%d%H%M}')

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
    print('  --> Already found most recent run. Done')
    found = False
elif last['run'] == date_recorded_12hr_ago:
    print(f'  --> Looking for run from 6 hours ago: {date_recorded_6hr_ago}')
    model_run = date_6hr_ago
    found = look_for_file(date_6hr_ago)
elif last['run'] == date_recorded_6hr_ago:
    print(f'  --> Looking for most recent run: {date_recorded}')
    model_run = date_recorded
    found = look_for_file(date_recorded)
else:
    print(f'  --> Looking for most recent run: {date_recorded}')
    model_run = date_recorded
    found = look_for_file(date_recorded)


# Log the results, if any.
if isinstance(found, dt):
    print(f'--> Found the most recent run! -> {found:%Y%m%d%H%M}\n--> Update model_available.csv')
    with open('./model_available.csv','w') as ma_csv:
        columns = ['run','available']
        ma = csv.DictWriter(ma_csv,fieldnames=columns)

        ma.writeheader()
        for row in rows:
            ma.writerow(row)
        print(f"  --> Writing new: 'run':'{model_run:%Y%m%d%H00}','available':'{found:%Y%m%d%H%M}'")
        ma.writerow({'run':f'{model_run:%Y%m%d%H00}','available':f'{found:%Y%m%d%H%M}'})
    print('--> File updated.')

print('--> DONE.')
