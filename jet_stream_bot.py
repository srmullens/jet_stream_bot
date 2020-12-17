########################################################################
#                                                                      #
# This code takes GFS data, identifies the jet stream and jet streaks, #
# and plots the ageostrophic components of wind along and across       #
# the calculated geostrophic wind. Divergence of wind, geostrophic     #
# wind, and components of ageostrophic wind is also calculated.        #
#                                                                      #
# Code by Stephen Mullens. May 2020.                                   #
#                                                                      #
########################################################################

from datetime import datetime as dt, timedelta
import time
import os
import copy

import metpy.calc as mpcalc
from metpy.units import units
from metpy.future import ageostrophic_wind
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker

# This is needed because of some error with cartopy and matplotlib axes 
# See also [Axes issue](https://github.com/SciTools/cartopy/issues/1120)
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import PIL
from twython import Twython


# IDEA: Can I find the 1x1 degree average of 850mb vorticity,
# contour the 1x10^5 (or whatever the exponent is) line,
# but weight the thickness of the line by the maximum
# average vorticity inside the contour? The higher the vorticity,
# the stronger the storm, the bolder the line.
# This may involve turning the contour into a polygon, or determining
# if a pixel is inside the polygon, to find the max vorticity.
#
# BONUS: Can I identify whether this is an extratropical or tropical
# storm? Can I label the lowest surface pressure with the appropriate
# symbol? Use the methodology in Philippe Papin's Climatology of 
# Central American Gyres paper.



####################
# MAP COMBINATIONS #
####################
# Speeds: real, geo, ageo
# Winds: real, geo, ageo, ageo_along, ageo_perp
# Speeds: real, geo, ageo, ageo_along, ageo_perp

# 1: real, real, real
# 2: geo, geo, geo
# 3: ageo, ageo, ageo
# 4: real, ageo_along, ageo_along
# 5: real, ageo_perp, ageo_perp
# 6: real, ageo_along, ageo_along_div
# 7: real, ageo_perp, ageo_perp_div
# 8: real, ageo, ageo_div

plots = [
        {'name':'Wind','contour_wind':'real','plot_barbs':'real','grid_fill':'real'},
        {'name':'Geostrophic wind','contour_wind':'geo','plot_barbs':'geo','grid_fill':'geo'},
        {'name':'Ageostrophic wind','contour_wind':'ageo','plot_barbs':'ageo','grid_fill':'ageo'},
        {'name':'Supergeostrophic Wind','contour_wind':'real','plot_barbs':'ageo_along','grid_fill':'ageo_along'},
        {'name':'Perpendicular Ageostrophic Wind','contour_wind':'real','plot_barbs':'ageo_perp','grid_fill':'ageo_perp'},
        {'name':'Divergence and Supergeostrophic Wind','contour_wind':'real','plot_barbs':'ageo_along','grid_fill':'ageo_along_div'},
        {'name':'Divergence and Perpendicular Ageostrophic Wind','contour_wind':'real','plot_barbs':'ageo_perp','grid_fill':'ageo_perp_div'},
        {'name':'Divergence and Wind','contour_wind':'real','plot_barbs':'real','grid_fill':'ageo_div'}
       ]


###########################################
# Settings for calculations and plotting. #
###########################################
# List of pressure levels to plot.
levels = [200, 250, 300, 400]
#levels = [200]
focus_level = 250

# Do you want height contours?
plot_hghts = True
plot_850_hghts = False

# smoothing: number of times (passes) data is smoothed
num_passes = 40

# Plot wind barbs or quivers every X grid places.
spacing = 10

# Barbs or Quiver?
barb_quiver = 'barb'

# Where do you want the map to be plotted?
#   'CONUS', 'Tropics', 'Carrib'
location = 'CONUS'

# Do you want to replace max colorbar values if
# higher wind speed or divergence is found?
replace = False

# Use forecast data? False uses analysis data.
forecast = True

# Use smoothed data or raw data?
use_smooth = True
if not use_smooth: replace = False

# Plot red where divergence values have been changed?
red_on_map = False

# Send tweet, or no?
send_tweet = False

# What forecast hour do you want to plot?
if forecast: fhr_list = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48] #,54,60,66,72,78,84,90,96]
else: fhr_list = [0]



######################################################################



#   ********************************************************************************
#   * Organization:                                                                *
#   * There's the MAIN script, which is at the bottom.                             *
#   * There are four PRIMARY functions, accessed by the main script.               *
#   * There are SECONDARY functions, mostly accessed by the primary make_images(). *
#   * There are TERTIARY functions, which support the secondary functions.         *
#   ********************************************************************************



########################################################
#                                                      # 
# TERTIARY Functions                                   #
# These are accessed by the secondary functions below. #
#                                                      # 
########################################################

# The divergence components suffer from artifacts where the wind speed
# is low but turns sharply. The turning of the wind fights with the
# Cartesian coordinate system. This function reduces the magnitude of
# the divergence values there, essentially removing those artifacts.
def adjust_div_values(lat,lon,smooth_hght,aGEO_along_divergence,aGEO_perp_divergence,grid_fill):
    """
    Function parameters:
        lat = latitude array
        lon = longitude array
        smooth_hght = height values
        aGEO_along_divergence = divergence from ageostrophic wind component along geostrophic streamline
        aGEO_perp_divergence = divergence from ageostrophic wind component perpendicular to geostrophic streamline.
        grid_fill = data to color fill on the map.
    """

    # Calculate the north-south derivative of the pressure surface height.
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    #hght_first = mpcalc.first_derivative(smooth_hght, delta=dy)

    hght_first_y = mpcalc.first_derivative(smooth_hght, delta=dy)
    #hght_first = hght_first_y
    #"""
    hght_first_x = mpcalc.first_derivative(smooth_hght, delta=dx, axis=1)
    hght_first_x = hght_first_x.flatten()
    hght_first_y = hght_first_y.flatten()
    hght_first = hght_first_y*hght_first_y + hght_first_x*hght_first_x
    hght_first = hght_first_y/np.abs(hght_first_y)*np.sqrt(hght_first)
    #"""
    # Flatten the 2D arrays to 1D arrays.
    lon_flat = lon.values.flatten()
    lat_flat = lat.values.flatten()
    hght_first = hght_first.flatten()
    aGEO_stream_div = aGEO_along_divergence.magnitude.flatten()
    aGEO_perp_div = aGEO_perp_divergence.magnitude.flatten()

    # Find locations to highlight
    count = []
    for i,val in enumerate(aGEO_along_divergence.flatten()):
        if -0.0001<hght_first[i]<0.0001:

            count.append(i)

            # Adjust divergence components.
            if aGEO_stream_div[i]>0:
                # Stream pos; Perp neg; Stream less
                if aGEO_stream_div[i]<np.abs(aGEO_perp_div[i]):
                    aGEO_perp_div[i]+= np.abs(aGEO_stream_div[i])
                    aGEO_stream_div[i]=0
                # Stream pos; Perp neg; Perp less
                else:
                    aGEO_stream_div[i]-= np.abs(aGEO_perp_div[i])
                    aGEO_perp_div[i]=0
            else:
                # Stream neg; Perp pos; Stream less
                if np.abs(aGEO_stream_div[i])<aGEO_perp_div[i]:
                    aGEO_perp_div[i]-= np.abs(aGEO_stream_div[i])
                    aGEO_stream_div[i]=0
                # Stream neg; Perp pos; Perp less
                else:
                    aGEO_stream_div[i]+= np.abs(aGEO_perp_div[i])
                    aGEO_perp_div[i]=0

    print(f'adjust {len(count)} of {len(lon_flat)} points')

    # Convert 1D aGEO_stream_div back to 2D.
    if grid_fill in ['ageo_along_div']:
        use_grid = aGEO_stream_div.reshape(lon.shape)
        use_grid = use_grid * units('1 / second')
    # Convert 1D aGEO_perp_div back to 2D.
    elif grid_fill in ['ageo_perp_div']:
        use_grid = aGEO_perp_div.reshape(lon.shape)
        use_grid = use_grid * units('1 / second')

    # Return 2D array of the divergence component.
    return use_grid



##############################################################
# Find the maximum element everywhere in the grid of values. #
# Given multiple wind values or multiple divergence values.  #
##############################################################
def max_arrays(*args):
    """
    Function parameters:
        args = several arrays. Can be any number.
    """
    # Find the maximum of each array.
    list_of_arg_maxes = []
    for arg in args:
        if arg.units in ['meter / second']: arg=arg.to('kts')
        arg = np.absolute(arg.magnitude)
        arg = np.nanmax(arg)
        list_of_arg_maxes.append(arg)

    # Find the maximum of the recorded maximums.
    max_value = np.nanmax(list_of_arg_maxes)

    return max_value



##################################################
# Format the labels on the colorbar.             #
# Used for divergence. Labels are whole numbers. #
# Colorbar label will show they are all *10^-5.  #
##################################################
def fmt(x, pos):
    """
    Function parameters:
        x = Number
        pos = Not sure. The ticker.FuncFormatter may need it.
    """

    # Make sure it's 10.0, not 10.0000000001
    x = round(x,5)

    # if 2x10^-4, a=2, b=-4
    a,b = f'{x:.1e}'.split('e')

    # I want all numbers to be 10^(-5). Adjust as needed.
    exp = int(b)+5
    a = float(a)*(10**exp)
    b = int(b)-exp

    # Create the final string.
    sci = f'{a:.0f}'

    # If zero, return 0. Otherwise, return the string.
    if a==0.0 and b==0.0: return '0'
    else: return sci



##############################################
# Take wind data for barb or quiver plotting #
##############################################
def mask_wind(U,V,wspd,min_speed,spacing):
    """
    Function parameters:
        U = east-west component of wind
        V = north-south component of wind
        wspd = wind speed
        min_speed = threshold value. Mask wind below this value.
        spacing = how many grid spaces apart should barb/quiver be plotted?
    """

    # create masked wind speed
    w = mpcalc.wind_speed(U,V)
    w = w.to('kts')

    # Create mask of wspd where calculated w is below min_speed threshold.
    # Note: w and wspd will often be the same. But they don't have to be.
    mask = []
    for i,values in enumerate(U):
        create_mask = np.ma.masked_where(w[i] < min_speed*units('kts'), wspd[i])
        mask.append(np.ma.getmaskarray(create_mask))

    # Convert mask from list of numpy arrays to 
    # a numpy array containing numpy arrays. (i.e. two-dimensional array)
    mask = np.array(mask)

    # apply the mask to u and v components
    masked_u = np.ma.masked_array(U,mask=mask)
    masked_v = np.ma.masked_array(V,mask=mask)

    # apply spacing for barb/quiver
    masked_u = masked_u[0::spacing,0::spacing]
    masked_v = masked_v[0::spacing,0::spacing]

    return masked_u, masked_v





######################################################
#                                                    # 
# SECONDARY Functions                                #
# These are accessed by the primary functions below. #
#                                                    # 
######################################################

####################################
# Send a tweet.                    #
# Info sent here by tweet_images() #
####################################
def tweet(text, image, send_tweet, reply):
    """
    Function parameters:
        text = tweet text. 280 characters or less.
        image = string file path or list of files to attach
        send_tweet = True/False flag. Send the tweet or not?
        reply = True/False flag. Is tweet original or a reply?
    """

    if send_tweet:
        """
        consumer_key = os.environ.get('consumer_key')
        consumer_secret = os.environ.get('consumer_secret')
        access_token = os.environ.get('access_token')
        access_token_secret = os.environ.get('access_token_secret')
        """

        consumer_key = 'PBAAvdU44v2xAN93je8w9YDwb'
        consumer_secret = '65rHGBKOBGCSJr0t8NBy7wQ92dQAS5jYlhSd5hXhmNmyB3jSvr'
        access_token = '1257385501478322181-7tcOOd2JJTVlTX40QcBGvEqtcfKyOR'
        access_token_secret = 'poZAMuSOZ9sEtvoR06RoHfYTSxWHmvNrHypETPr2vwehv'

        print('  --> Tweeting...')
        twitter = Twython(consumer_key, consumer_secret, access_token, access_token_secret)

        # Tweet new status.
        response_list=()
        if not reply:
            # Assemble images
            if isinstance(image,list):
                for img in image:
                    response = twitter.upload_media(media=open(img, 'rb'))
                    response_list = (*response_list, response['media_id'])
            elif isinstance(image,str) and image[:-3]=='gif':
                response = twitter.upload_video(media = open(img, 'rb'),
                                                media_type = 'image/gif',
                                                media_category = 'tweet_gif')
                print(response['media_id'])
                response_list = (*response_list, response['media_id'])
            elif isinstance(image,str) and image[-3:] in ['png','jpg']:
                response = twitter.upload_media(media=open(img, 'rb'))
                response_list = (*response_list, response['media_id'])

            # Send the tweet
            twitter.update_status(status=text, media_ids=list(response_list))


        # Tweet a reply.
        elif reply:
            # ...Get most recent tweet's ID from the timeline...
            timeline = twitter.get_user_timeline(screen_name='jetstreambot',count=5)
            tweet_list = []
            for tweet in timeline:
                created = dt.strptime(tweet['created_at'],'%a %b %d %H:%M:%S %z %Y')
                tweet_list.append({'created_at':created,'id':tweet['id']})
                tweet_list = sorted(tweet_list, key = lambda i: i['created_at'],reverse=True)
                tweet_id = tweet_list[0]['id']

            # Assemble images
            if isinstance(image,list):
                for img in image:
                    response = twitter.upload_media(media=open(img, 'rb'))
                    response_list = (*response_list, response['media_id'])
            elif isinstance(image,str) and image[-3:]=='gif':
                response = twitter.upload_video(media = open(image, 'rb'),
                                                media_type = 'image/gif',
                                                media_category = 'tweet_gif')
                response_list = (*response_list, response['media_id'])
            elif isinstance(image,str) and image[-3:] in ['png','jpg']:
                response = twitter.upload_media(media=open(image, 'rb'))
                response_list = (*response_list, response['media_id'])

            # Send the tweet.
            twitter.update_status(status=text,
                                media_ids=list(response_list),
                                in_reply_to_status_id=tweet_id,
                                auto_populate_reply_metadata=True)


    # Show where the tweet would be sent.
    else:
        if not reply: print('    --> TEST Original tweet')
        elif reply: print('    --> TEST Reply to tweet')

    print('  --> Tweeted.')



#######################################
# Functions accessed by make_images() #
#######################################

###########################################
# Raise warnings if something won't work. #
###########################################
def problems(plots,level,num_passes,spacing,barb_quiver,plot_hghts):

    """
    Function parameters:
        plots = list of plots to make.
        level = pressure level in millibars or hectopascals.
        num_passes = number of times smoothing function is applied.
        spacing = how spaced out barbs or quiver arrows are in plots.
        barb_quiver = flag, either 'barb' or 'quiver'
        plot_hghts = True/False flag, plot height contours on map or not
    """

    # Make sure we know what we're plotting.
    for plot in plots:
        if plot['contour_wind'] not in ['real', 'geo', 'ageo']:
            warn = f"Asked to contour {plot['contour_wind']}.\nMust be 'real', 'geo', or 'ageo'"
            raise ValueError(warn)

        if plot['plot_barbs'] not in ['real', 'geo', 'ageo', 'ageo_along', 'ageo_perp']:
            warn = f"Asked to plot {plot['plot_barbs']} barbs.\nMust be 'real', 'geo', 'ageo', 'ageo_along', or 'ageo_perp'"
            raise ValueError(warn)

        if plot['grid_fill'] not in ['real', 'geo', 'ageo', 'ageo_along', 'ageo_perp', 'real_div', 'geo_div', 'ageo_div', 'ageo_along_div', 'ageo_perp_div']:
            warn = f"Asked to plot {plot['grid_fill']} grid.\nMust be 'real', 'geo', 'ageo', 'ageo_along', 'ageo_perp', 'real_div', 'geo_div', 'ageo_div', 'ageo_along_div', or 'ageo_perp_div'"
            raise ValueError(warn)

    # Make sure the variables are the right type.
    if not isinstance(level, (int, float)):
        warn = f"Asked to plot level with type {type(level)} ({level}). Must be an integer or float."
        raise TypeError(warn)
    elif level not in [50,100,150,200,250,300,400,500,600,700,800,850,925,1000]:
        warn = f"Asked to plot level {level}. Must be a standard level in hPa."
        raise ValueError(warn)

    if not isinstance(num_passes, int):
        warn = f"num_passes must be an integer. You put a {type(num_passes)}."
        raise TypeError(warn)

    if not isinstance(spacing, int):
        warn = f"spacing must be an integer. You put a {type(spacing)}."
        raise TypeError(warn)

    if barb_quiver not in ['barb','quiver']:
        warn = f"Asked to plot 'barb' or 'quiver', but you put barb_quiver={barb_quiver}."
        raise ValueError(warn)

    if not isinstance(plot_hghts, bool):
        warn = f"plot_heights must be either True or False. You put {type(plot_hghts)}"
        raise TypeError(warn)

    # Make a folder for the level if one doesn't already exist.
    if not os.path.exists(f'{level}'):
        os.makedirs(f'{level}')

    return True



#################################################
# Using the data, calculate the fields to plot. #
#################################################
def calculate_variables(data=None, date=None, forecast=False, fhr=0, level=200, num_passes=40, loc='CONUS'):

    # How long will it take to get the data?
    start_time = dt.now()

    """
    Function parameters:
        data = gathered data
        date = datetime object
        forecast = False to use analysis data, True if using forecast files.
        fhr = forecast hour(s).
        level = pressure level for the plot in hectopascals or millibars.
                    Typically in the jet stream, so 200hPa down to 400hPa.
        num_passes = controls how smooth the plot is.
                    40 passes means data 10 degrees away has an influence here.
        loc = location to plot. 'CONUS', 'Tropics', 'Carrib'
    """


    ####
    # If date not provided, get date.
    ####
    if date is None:
        print(f"--> Using today.")
        # Set date for desired UPA data
        today = dt.utcnow()

        # Most recent UTC hour.
        if today.hour<6: hour = 0
        elif today.hour<12: hour = 6
        elif today.hour<18: hour = 12
        else: hour = 18

        # Go back one synoptic time to ensure data availability
        date = dt(today.year, today.month, today.day, hour) - timedelta(hours=6)

    # If date is provided, use it.
    elif isinstance(date, dt):
        print(f"--> Using given date: {date:%Y-%m-%d %H UTC}")
        date = date

    # If date is provided by isn't a datetime instance, raise warning.
    else:
        warn=f'Date is type {type(date)}. Must be type datetime.datetime.'
        raise TypeError(warn)


    ####
    # If data is not provided, get forecast data.
    #   Obtain GFS gridded output for contour plotting. Specifically,
    #   geopotential height and temperature data for the given level and subset
    #   for over North America. Data are smoothed for aesthetic reasons.
    ####

    if data is None:
        print("--> Data not given! Getting data...")
        # Get GFS data and subset to North America for Geopotential Height and Temperature
        ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg_ana/'
                             f'GFS_Global_0p25deg_ana_{date:%Y%m%d}_{date:%H}00.grib2').metpy.parse_cf(['Geopotential_height_isobaric','u-component_of_wind_isobaric','v-component_of_wind_isobaric'])
        print("--> Got data.")

    else:
        print("--> Already have data")
        ds = data


    # Get data from the correct time.
    #   Set the plot time with forecast hours
    if forecast: date = date + timedelta(hours=18)


    ####
    # Get and Smooth Basic Parameters
    ####

    # Parameters of the location we're looking for.
    if loc == 'CONUS':
        location = {'vertical':level*units.hPa,
                'time':date+timedelta(hours=fhr),
                'lat':slice(70, 15),
                'lon':slice(360-145, 360-50)
        }
        location_850 = {'vertical':850*units.hPa,
                'time':date+timedelta(hours=fhr),
                'lat':slice(70, 15),
                'lon':slice(360-145, 360-50)
        }
    elif loc == 'Tropics':
        location = {'vertical':level*units.hPa,
                'time':date+timedelta(hours=fhr),
                'lat':slice(50, 1),
                'lon':slice(360-120, 360-5)
        }
        location_850 = {'vertical':850*units.hPa,
                'time':date+timedelta(hours=fhr),
                'lat':slice(50, 1),
                'lon':slice(360-120, 360-5)
        }
    elif loc == 'Carrib':
        location = {'vertical':level*units.hPa,
                'time':date+timedelta(hours=fhr),
                'lat':slice(50, 5),
                'lon':slice(360-120, 360-45)
        }
        location_850 = {'vertical':850*units.hPa,
                'time':date+timedelta(hours=fhr),
                'lat':slice(50, 5),
                'lon':slice(360-120, 360-45)
        }

    # Get geopotential height data for the specified pressure level.
    hght = ds['Geopotential_height_isobaric'].metpy.loc[location]
    smooth_hght = mpcalc.smooth_n_point(hght, 9, num_passes)
    # Smooth the height data
    if use_smooth: use_hght = smooth_hght
    else: use_hght = hght

    # Get U and V wind component data
    uwind = ds['u-component_of_wind_isobaric'].metpy.loc[location]
    vwind = ds['v-component_of_wind_isobaric'].metpy.loc[location]
    uwind = mpcalc.smooth_n_point(uwind, 9, num_passes)
    vwind = mpcalc.smooth_n_point(vwind, 9, num_passes)

    # Get geographical height data for 850mb
    hght_850 = ds['Geopotential_height_isobaric'].metpy.loc[location_850]
    smooth_hght_850 = mpcalc.smooth_n_point(hght_850, 9, num_passes)
    use_hght_850 = hght_850

    # Get U and V wind component data for 850mb
    uwind_850 = ds['u-component_of_wind_isobaric'].metpy.loc[location_850]
    vwind_850 = ds['v-component_of_wind_isobaric'].metpy.loc[location_850]
    uwind_850 = mpcalc.smooth_n_point(uwind_850, 9, num_passes)
    vwind_850 = mpcalc.smooth_n_point(vwind_850, 9, num_passes)


    ####
    # Manipulate the wind data
    #   Calculate the Coriolis (f), Geostrophic (GEO), and Ageostrophic (aGEO) wind
    #   Calculate the divergence of the real, GEO, and aGEO winds.
    ####

    # Get the mapping crs from the data.
    data_crs = hght.metpy.cartopy_crs

    # Get the x, y, lat, and lon lists from the data.
    x, y = hght.metpy.coordinates('x','y')
    lat, lon = xr.broadcast(y, x)

    # Use lats and lons to get change in x and y in meters.
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat, initstring=data_crs.proj4_init)

    # Let MetPy calculate the Coriolis parameter.
    f = mpcalc.coriolis_parameter(lat)

    # Let MetPy calculate the geostrophic and ageostrophic wind components.
    uGEO, vGEO = mpcalc.geostrophic_wind(use_hght, f, dx, dy, dim_order='yx')
    uaGEO, vaGEO = ageostrophic_wind(use_hght, uwind, vwind, f, dx, dy, dim_order='yx')

    # Let MetPy calculate the vertical component of the relative vorticity at 850mb.
    vort_850 = mpcalc.vorticity(uwind_850, vwind_850, dx, dy)


    ####
    # Use GEO and aGEO to calculate aGEO components
    #   Ccalculate GEO unit vectors,
    #   direction of GEO wind and perpendicular to GEO wind,
    #   aGEO wind components along and perpendicular to GEO wind,
    #   the signed direction of aGEO components relative to GEO wind,
    #   and the divergence from aGEO components relative to GEO wind.
    ####

    # Let MetPy calculate actual, geostrophic, and ageostrophic wind speeds.
    wspd = mpcalc.wind_speed(uwind,vwind)
    GEOspd = mpcalc.wind_speed(uGEO,vGEO)
    aGEOspd = mpcalc.wind_speed(uaGEO,vaGEO)

    # Calculate u and v unit vector components of GEO.
    uGEO_unit = uGEO / GEOspd
    vGEO_unit = vGEO / GEOspd


    ####
    # Calculate the magnitude of aGEO wind along and perpendicular
    # to the GEO unit vector.
    #   unit_vector_angle = angle of GEO unit vector counter-clockwise from eastward direction.
    #   perp_vector_angle = 90 degrees counter-clockwise from unit_vector_angle.
    #
    #   aGEO_along_GEO = magnitude of aGEO wind directly with/against GEO wind.
    #   aGEO_perp_GEO = magnitude of aGEO wind perpendicular to GEO wind.
    ####

    unit_vector_angle = np.arctan2(vGEO_unit,uGEO_unit)
    perp_vector_angle = unit_vector_angle + (np.pi/2)*units('radian')

    aGEO_along_GEO = uaGEO*np.cos(unit_vector_angle) + vaGEO*np.sin(unit_vector_angle)
    aGEO_perp_GEO = -uaGEO*np.sin(unit_vector_angle) + vaGEO*np.cos(unit_vector_angle)


    ####
    # Preserve where the aGEO components are postiive or negative.
    #   Divide the number by its magnitude. Thus, if value >=0.0: =1; else: =-1
    ####

    #   Upstream is negative. Downstream is positive.
    direction_along_GEO = aGEO_along_GEO / np.absolute(aGEO_along_GEO)
    #   Left is positive. Right is negative.
    direction_perp_GEO = aGEO_perp_GEO / np.absolute(aGEO_perp_GEO)


    ####
    # Convert those magnitudes back to normal x/y coordinates.
    #   aGEO_along_u = E-W component of aGEO wind in direction of GEO wind
    #   aGEO_along_v = N-S component of aGEO wind in direction of GEO wind
    #
    #   aGEO_perp_u = E-W component of aGEO wind perpendicular to GEO wind
    #   aGEO_perp_v = N-S component of aGEO wind perpendicular to GEO wind
    ####

    aGEO_along_u = aGEO_along_GEO * np.cos(unit_vector_angle)
    aGEO_along_v = aGEO_along_GEO * np.sin(unit_vector_angle)
    # compontents to speed with +/- sign
    aGEO_along_spd = mpcalc.wind_speed(aGEO_along_u,aGEO_along_v)
    aGEO_along_spddir = aGEO_along_spd * direction_along_GEO

    aGEO_perp_u = aGEO_perp_GEO * np.cos(perp_vector_angle)
    aGEO_perp_v = aGEO_perp_GEO * np.sin(perp_vector_angle)
    # compontents to speed with +/- sign
    aGEO_perp_spd = mpcalc.wind_speed(aGEO_perp_u,aGEO_perp_v)
    aGEO_perp_spddir = aGEO_perp_spd * direction_perp_GEO



    ####
    # Calculate the divergence of all the parameters.
    ####

    # While we're here, calculate wind, GEO, aGEO divergences
    wind_divergence = mpcalc.divergence(uwind,vwind,dx,dy)
    GEO_divergence = mpcalc.divergence(uGEO,vGEO,dx,dy)
    aGEO_divergence = mpcalc.divergence(uaGEO,vaGEO,dx,dy)

    # Calculate the divergence of the aGEO components along and perpendicular
    # to the GEO wind.
    #   aGEO wind along/against GEO wind is super- and sub-geostrophic wind.
    #   aGEO wind across GEO wind is 4-quadrant model of jet streaks.
    aGEO_along_divergence = mpcalc.divergence(aGEO_along_u,aGEO_along_v,dx,dy)
    aGEO_perp_divergence  = mpcalc.divergence(aGEO_perp_u,aGEO_perp_v,dx,dy)



    ####
    # Find 1-degree or 1000km average of 850mb vorticity.
    #   Later: Add 850mb vort to args_div
    #           Add 1x10-4 contour of vort_850 to plot.
    #           Adjust contour thickness by max amount.
    #           Show min sfc pressure location inside vort_850 contour.
    #           Figure something out for Highs
    ####




    ####
    # Trim the data to remove pixels around the edge of the region.
    ####

    lon = lon[5:-5,5:-5]
    lat = lat[5:-5,5:-5]

    smooth_hght = use_hght[5:-5,5:-5]
    smooth_hght_850 = use_hght_850[5:-5,5:-5]

    wspd = wspd[5:-5,5:-5]
    GEOspd = GEOspd[5:-5,5:-5]
    aGEOspd = aGEOspd[5:-5,5:-5]
    aGEO_along_spddir = aGEO_along_spddir[5:-5,5:-5]
    aGEO_perp_spddir = aGEO_perp_spddir[5:-5,5:-5]

    wind_divergence = wind_divergence[5:-5,5:-5]
    GEO_divergence = GEO_divergence[5:-5,5:-5]
    aGEO_divergence = aGEO_divergence[5:-5,5:-5]
    aGEO_along_divergence = aGEO_along_divergence[5:-5,5:-5]
    aGEO_perp_divergence = aGEO_perp_divergence[5:-5,5:-5]

    """
    # Output values to make sure things look right.
    # .2f = plot numbers to 2 decimal points.
    # ~P = "pretty print" abbreviated units.

    print("\nMax speeds:")
    print(f"wind:\t{np.nanmax(wspd):.2f~P}")
    print(f"GEO:\t{np.nanmax(GEOspd):.2f~P}")
    print(f"aGEO:\t{np.nanmax(aGEOspd):.2f~P}")
    print(f"aGEO_along_spd:\t{np.nanmax(aGEO_along_spddir):.2f~P}")
    print(f"aGEO_perp_spd:\t{np.nanmax(aGEO_perp_spddir):.2f~P}")

    print("\nDivergences:")
    print(f"wind:\t{np.nanmax(wind_divergence):.2e~P}\t{np.nanmin(wind_divergence):.2e~P}")
    print(f"GEO:\t{np.nanmax(GEO_divergence):.2e~P}\t{np.nanmin(GEO_divergence):.2e~P}")
    print(f"aGEO:\t{np.nanmax(aGEO_divergence):.2e~P}\t{np.nanmin(aGEO_divergence):.2e~P}")
    print(f"aGEO_along:\t{np.nanmax(aGEO_along_divergence):.2e~P}\t{np.nanmin(aGEO_along_divergence):.2e~P}")
    print(f"aGEO_perp:\t{np.nanmax(aGEO_perp_divergence):.2e~P}\t{np.nanmin(aGEO_perp_divergence):.2e~P}")
    """

    # Package lists into fewer variables.
    args_map = (lat,lon,smooth_hght,smooth_hght_850)
    args_uv = (uwind,vwind, uGEO,vGEO, uaGEO,vaGEO, aGEO_along_u,aGEO_along_v, aGEO_perp_u,aGEO_perp_v)
    args_spd = (wspd, GEOspd, aGEOspd, aGEO_along_spddir, aGEO_perp_spddir)
    args_div = (wind_divergence, GEO_divergence, aGEO_divergence, aGEO_along_divergence, aGEO_perp_divergence)



    # Time to get data
    time_elapsed = dt.now() - start_time
    tsec = round(time_elapsed.total_seconds(),2)
    print(f"--> Variables calculated ({tsec:.2f} seconds)")


    return args_map, args_uv, args_spd, args_div



##################################################################
# Get the maximum values for the colorbars from a file.          #
# Make sure the current values don't exceed those from the file. #
# If they do, replace the file.                                  #
##################################################################
def get_bounds(colorbar_maxes, wspd, GEOspd, aGEOspd, aGEO_along_spddir, aGEO_perp_spddir, wind_divergence, GEO_divergence, aGEO_divergence, aGEO_along_divergence, aGEO_perp_divergence, loc='CONUS', replace=True):

    # Compare to max values. See if any of them need to be replaced.
    restart_if_false = True

    # If you find a max value bigger than what's been recorded, restart the loop of plots.
    if colorbar_maxes is not None:
        replace1 = float(colorbar_maxes[0])
        replace2 = float(colorbar_maxes[1])
        replace3 = float(colorbar_maxes[2])
        print(f"--> old: {replace1:.2f}, {replace2:.2f}, {replace3:.2e}")
        this1 = max_arrays(wspd,GEOspd,aGEOspd)
        this2 = max_arrays(aGEO_along_spddir,aGEO_perp_spddir)
        this3 = max_arrays(wind_divergence,GEO_divergence,aGEO_divergence)
        max1 = max2 = max3 = False

        print(f"--> Max: {this1:.2f}, {this2:.2f}, {this3:.2e}")

        if not replace: return restart_if_false,replace1,replace2,replace3
        else:

            if replace1 < this1:
                print(f"XXX\nXXX max wind speed colorbar value replaced.\nXXX OLD: {replace1}, NEW: {this1}\nXXX")
                replace1 = this1
                restart_if_false = False
                max1 = True

            if replace2 < this2:
                print(f"XXX\nXXX max 2-sided wind speed colorbar value replaced.\nXXX OLD: {replace2}, NEW: {this2}\nXXX")
                replace2 = this2
                restart_if_false = False
                max2 = True

            if replace3 < this3:
                print(f"XXX\nXXX max divergence colorbar value replaced.\nXXX OLD: {replace3}, NEW: {this3}\nXXX")
                replace3 = this3
                restart_if_false = False
                max3 = True

            # Replace the file of max values.
            if np.isinf(np.any([max1,max2,max3])):
                warn=f'Wind and/or divergences contain infinite values.'
                raise TypeError(warn)
            elif not replace: pass
            elif replace and np.any([max1,max2,max3]):
                os.rename(f'./{loc}_max_colorbar_values.txt',f'./{loc}_max_colorbar_values_old.txt')
                header = 'one_sided_wind,two_sided_wind,divergence\n'
                replace = f"{replace1},{replace2},{replace3}"

                with open(f'./{loc}_max_colorbar_values.txt','w') as file_of_maxes:
                    file_of_maxes.write(header)
                    file_of_maxes.write(replace)
                    file_of_maxes.close()

            return restart_if_false,replace1,replace2,replace3



########################################################################
# Make a scatter plots revealing how divergence data is being altered. # 
########################################################################
def diagnostic_scatter_plot(level,fhr,args_div,args_map,args_uv):

    # How long will it take to make the plots?
    start_time = dt.now()

    # Unpack the data to plot.
    wind_divergence, GEO_divergence, aGEO_divergence, aGEO_along_divergence, aGEO_perp_divergence = args_div
    lat,lon,smooth_hght,smooth_hght_850 = args_map
    uwind,vwind, uGEO,vGEO, uaGEO,vaGEO, aGEO_along_u,aGEO_along_v, aGEO_perp_u,aGEO_perp_v = args_uv

    # Calculate speed of real wind.
    # Remove outermost pixels.
    wspd = mpcalc.wind_speed(uwind, vwind).to('kts')
    wspd = wspd[5:-5,5:-5]

    # Calculate derivative of height from equator to pole.
    # dx, dy = distance between pixels.
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    hght_first_y = mpcalc.first_derivative(smooth_hght, delta=dy)
    #hght_first = hght_first_y

    hght_first_x = mpcalc.first_derivative(smooth_hght, delta=dx, axis=1)
    hght_first_x = hght_first_x.flatten()
    hght_first_y = hght_first_y.flatten()
    hght_first = hght_first_y*hght_first_y + hght_first_x*hght_first_x
    hght_first = hght_first_y/np.abs(hght_first_y)*np.sqrt(hght_first)

    # Flatten 2D arrays to 1D array
    aGEO_stream_div = copy.deepcopy(aGEO_along_divergence).magnitude.flatten()
    aGEO_perp_div = copy.deepcopy(aGEO_perp_divergence).magnitude.flatten()
    div = aGEO_divergence.flatten()

    if isinstance(smooth_hght, xr.DataArray): smooth_hght = smooth_hght.metpy.unit_array
    smooth_hght = smooth_hght.flatten()
    hght_first = hght_first.flatten()
    wspd = wspd.flatten()

    # Plot in red the data where the divergence values must be adjusted to remove artifacts.
    #   Where height derivative is small, reduce divergence component with smallest
    #   magnitude to zero. Reduce other divergence component by equivalent amount. If this
    #   second component is still great enough to be colored in the plot, so be it.
    same_wspd = []
    same_stream_mag = []
    same_perp_mag = []
    same_div = []
    same_hght_first = []
    same_hght = []
    for i,val in enumerate(aGEO_stream_div):
        if -0.0001<hght_first[i]<0.0001:

            if aGEO_stream_div[i]>0:
                # Stream pos; Perp neg; Stream less
                if aGEO_stream_div[i]<np.abs(aGEO_perp_div[i]):
                    aGEO_perp_div[i]+= np.abs(aGEO_stream_div[i])
                    aGEO_stream_div[i]=0
                # Stream pos; Perp neg; Perp less
                else:
                    aGEO_stream_div[i]-= np.abs(aGEO_perp_div[i])
                    aGEO_perp_div[i]=0
            else:
                # Stream neg; Perp pos; Stream less
                if np.abs(aGEO_stream_div[i])<aGEO_perp_div[i]:
                    aGEO_perp_div[i]-= np.abs(aGEO_stream_div[i])
                    aGEO_stream_div[i]=0
                # Stream neg; Perp pos; Perp less
                else:
                    aGEO_stream_div[i]+= np.abs(aGEO_perp_div[i])
                    aGEO_perp_div[i]=0

            # These are the data that will be plotted red.
            same_wspd.append(wspd[i])
            same_stream_mag.append(np.abs(aGEO_stream_div[i]))
            same_perp_mag.append(np.abs(aGEO_perp_div[i]))
            same_div.append(div[i])
            same_hght_first.append(hght_first[i])
            same_hght.append(smooth_hght[i])

    print(f'red is {len(same_wspd)} of {len(aGEO_stream_div)} points')

    # Make subplots
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)

    kwargs = {"c":"k","s":1}

    # Divergence magnitude vs wind speed.
    ax1.scatter(wspd,np.abs(aGEO_along_divergence.magnitude.flatten()),c='gray',s=1)
    ax1.scatter(wspd,-np.abs(aGEO_perp_divergence.magnitude.flatten()),c='gray',s=1)
    ax1.scatter(wspd,np.abs(aGEO_stream_div),**kwargs)
    ax1.scatter(wspd,-np.abs(aGEO_perp_div),**kwargs)
    ax1.scatter(same_wspd,same_stream_mag,c='red',s=1)
    ax1.scatter(same_wspd,[-i for i in same_perp_mag],c='red',s=1)

    ax1.set_xlabel(f'wind speed (knot)')
    ax1.set_ylabel('stream <--> perp\ngray=old, black=new, red=adjusted')
    ax1.set_xlim(left=0,right=60)
    ax1.set_ylim(bottom=-0.0002,top=0.0002)
    ax1.axhline(y=-0.00002)
    ax1.axhline(y=0.00002)
    ax1.set(adjustable='box')

    # Height vs Wind speed
    ax2.scatter(wspd,smooth_hght,**kwargs)
    ax2.scatter(same_wspd,same_hght,c='red',s=1)

    ax2.set_xlabel('wind speed (knot)')
    ax2.set_ylabel('height')
    ax2.set_xlim(left=0) #,right=60)
    ax2.set_ylim(bottom=np.min(smooth_hght.magnitude)-50,top=np.max(smooth_hght.magnitude)+50)
    ax2.axvline(x=0)
    ax2.set(adjustable='box')

    # Height derivative vs Height
    ax3.scatter(hght_first,smooth_hght,**kwargs)
    ax3.scatter(same_hght_first,same_hght,c='red',s=1)

    ax3.set_xlabel('hght_first_derivative')
    ax3.set_ylabel('height')
    ax3.set_xlim(left=-0.001,right=0.001)
    ax3.set_ylim(bottom=np.min(smooth_hght.magnitude)-50,top=np.max(smooth_hght.magnitude)+50)
    ax3.axvline(x=0)
    ax3.set(adjustable='box')

    # Wind speed vs Height derivative
    ax4.scatter(wspd,hght_first,**kwargs)
    ax4.scatter(same_wspd,same_hght_first,c='red',s=1)

    ax4.set_xlabel(f'wind speed (knot)')
    ax4.set_ylabel('hght_first_derivative')
    ax4.set_ylim(bottom=-0.001,top=0.001)
    ax4.axhline(y=0)
    ax4.set(adjustable='box')


    print("--> Saving graph")

    fig.tight_layout(pad=1.0)
    fig.savefig(f"{level}/{level}_{fhr}_graphs.png")

    # Clear the axis.
    plt.close()

    # Time to plot data
    time_elapsed = dt.now() - start_time
    tsec = round(time_elapsed.total_seconds(),2)
    print(f"--> Plotted graphs ({tsec:.2f} seconds)")



##########################################
# Plot maps using data from make_image() #
##########################################
def plot_the_map(args_map,args_uv,args_spd,args_div,date,
            contour_wind='real', plot_barbs='real', grid_fill='real', name=None, fhr=0,
            level=200,
            bounds_set=None,
            spacing=10, barb_quiver='barb', plot_hghts=True, loc='CONUS', red_on_map=False):

    # How long will it take to get the data?
    start_time = dt.now()

    """
    Function parameters:
        args_map = lat, lon, height, 850hPa data for plotting.
        args_uv = u and v wind components for plotting barbs
        args_spd = wind speed values for plotting grids
        args_div = divergence values for plotting grids
        date = datetime variable of plotting time
        contour_wind = real, geo, ageo
                    Blue contours of isotachs of the wind field. (kts)
        plot_barbs = real, geo, ageo, ageo_along, ageo_perp
                    Black wind barbs of the wind field. (kts)
        grid_fill = Can plot wind speed or divergence in the field.
                    Speeds: real, geo, ageo, ageo_along, ageo_perp
                    Divergences: real_div, geo_div, ageo_div, ageo_along_div, ageo_perp_div
                        Real, geo, ageo winds are plotted using white to blue colormap.
                        All others are plotted using custom red/blue two-sided colormap.
        name = Name for the title of the plot.
        fhr = How many forecast hours in the future to plot.
        level = pressure level for the plot in hectopascals or millibars.
                    Typically in the jet stream. Used in title of the plot.
        bounds_set = set of extreme values to use when plotting grid colorbars.
        spacing = plots barbs every x grid points. Default is 10.
        barb_quiver = plots either barbs (default) or quivers (aka arrows).
        plot_hghts = Include the black height contours or not.
        loc = 'CONUS', 'Tropics', 'Carrib' Used to set [W, E, S, N] extent where plot is located.
        red_on_map = Plots red areas where divergence values have been altered.
    """

    # Unpack the data to plot
    lat,lon,smooth_hght,smooth_hght_850 = args_map
    uwind,vwind, uGEO,vGEO, uaGEO,vaGEO, aGEO_along_u,aGEO_along_v, aGEO_perp_u,aGEO_perp_v = args_uv
    wspd, GEOspd, aGEOspd, aGEO_along_spddir, aGEO_perp_spddir = args_spd
    wind_divergence, GEO_divergence, aGEO_divergence, aGEO_along_divergence, aGEO_perp_divergence = args_div
    wind_1d_bound, wind_2d_bound, div_bound = bounds_set


    ###
    # Set minimum and maximum values for plotting contours, grids, barbs.
    # -------------------------
    # Because of above settings...
    #    ...set these settings.
    ###

    # What minimum value wind speed do you want to contour? (knots)
    if contour_wind in ['real','geo']:
        if level < 400: min_contour = 75
        else:           min_contour = 50
    elif contour_wind in ['ageo']:
        min_contour = 25

    # What minimum value do you want barbs or quivers plotted for? (knots)
    if plot_barbs in ['real','geo']:
        if level < 400: min_barb = 75
        else:           min_barb = 50
    elif plot_barbs in ['ageo']:
        min_barb = 15
    elif plot_barbs in ['ageo_along','ageo_perp']:
        min_barb = 10

    # What minimum value do you want the grid fill to be colored in?
    if grid_fill in ['real','geo']:
        if level < 400: min_grid = 50
        else:           min_grid = 40
    elif grid_fill in ['ageo']:
        min_grid = 15
    elif grid_fill in ['ageo_along','ageo_perp']:
        min_grid = 10
    elif grid_fill in ['real_div','geo_div','ageo_div','ageo_along_div','ageo_perp_div']:
        if level < 400: min_grid = 0.00002
        else:           min_grid = 0.000015


    ###
    # Format contour intervals and labels
    # -------------------------
    # The contour interval (cint) is set based on the level chosen.
    # If you want to label the contours, the label format is also given.
    ###

    if (level == 925) | (level == 850) | (level == 700):
        cint = 30
        def hght_format(v): return format(v, '.0f')[1:]
    elif level == 500:
        cint = 60
        def hght_format(v): return format(v, '.0f')[:3]
    elif level == 400:
        cint = 60
        def hght_format(v): return format(v, '.0f')[:3]
    elif level == 300:
        cint = 120
        def hght_format(v): return format(v, '.0f')[:3]
    elif level < 300:
        cint = 120
        def hght_format(v): return format(v, '.0f')[1:4]


    ###
    # Set map extent based on location to plot.
    ###
    if loc in ['CONUS']:
        extent=[-125, -70, 20, 55]
        mapcrs = ccrs.Orthographic(central_latitude=39.833333, central_longitude=-98.583333)
    elif loc in ['Tropics']:
        extent=[-95, -15, 5, 35]
        mapcrs = ccrs.Orthographic(central_latitude=5, central_longitude=-50)
    elif loc in ['Carrib']:
        extent=[-100, -55, 5, 35]
        mapcrs = ccrs.Orthographic(central_latitude=23, central_longitude=-82)
    # If none identified, show CONUS.
    elif loc is None:
        extent=[-125, -70, 20, 55]
        mapcrs = ccrs.Orthographic(central_latitude=39.833333, central_longitude=-98.583333)
    # If list of [min_lon, max_lon, min_lat, max_lat] given, plot that.
    elif isinstance(loc, list) and len(loc)==4:
        extent=loc
        clat = (extent[2]+extent[3])/2
        clon = (extent[0]+extent[1])/2
        mapcrs = ccrs.Orthographic(central_latitude=clat, central_longitude=clon)
    else:
        warn = f'Location must be a CONUS, Tropics, Carrib, or list of length four: [W, E, S, N]\nYou gave {type(loc)}, length {len(loc)}, {loc}.'
        raise RuntimeError(warn)

    ###########################
    # Start making the figure #
    ###########################
    fig1 = plt.figure(1, figsize=(16, 9), dpi=180)
    ax1 = plt.subplot(111, projection=mapcrs)

    # Set extent
    ax1.set_extent(extent)

    # Add map features for geographic reference
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey')
    ax1.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white')
    ax1.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey')


    ######
    # Plot Solid Contours of Geopotential Height
    ######
    # Plot height of current pressure level.
    if plot_hghts:
        kwargs = {'colors':'black',
                'linewidths':1,
                'levels':range(0, 20000, cint),
                'transform':ccrs.PlateCarree()
                }
        cs = ax1.contour(lon, lat, smooth_hght, **kwargs)

        legend1,_ = cs.legend_elements()
        label1 = f'{level}hPa Heights\n(every {cint}m)'
    else: pass

    # Plot 850mb heights, if desired.
    if plot_850_hghts:
        kwargs = {'colors':'gray',
                'linewidths':0.5,
                'levels':range(0, 20000, 30),  # contour every 30m
                'transform':ccrs.PlateCarree()
                }
        cs850 = ax1.contour(lon, lat, smooth_hght_850, **kwargs)

        legend850,_ = cs850.legend_elements()
        label850 = f'850hPa Heights\n(every {30}m)'
    else: pass


    ######
    # Plot contour of wind speed
    ######

    # Which wind are we plotting? Grab that data and legend name.
    if contour_wind in ['real']:
        use_wind = wspd
        lgd = 'Wind'
    elif contour_wind in ['geo']:
        use_wind = GEOspd
        lgd = 'Geostrophic Wind'
    elif contour_wind in ['ageo']:
        use_wind = aGEOspd
        lgd = 'Ageostrophic Wind'
    elif contour_wind is None:
        use_wind==None

    # Plot isotachs
    if use_wind is not None:
        # Mask the wind speed below a threshold.
        masked_wspd = []
        for i,w in enumerate(use_wind):
            w = w.to('kts')
            masked_w = np.ma.masked_where(w < min_contour*units('kts'), w)
            masked_wspd.append(masked_w)
        masked_wspd = masked_wspd*units('kts')

        # Contour wind speed isotachs
        kwargs = {'colors':'blue',
                    'levels':range(min_contour,int(np.nanmax(masked_wspd.magnitude)+1),25),
                    'transform':ccrs.PlateCarree()
                }
        cspd = ax1.contour(lon, lat, masked_wspd, **kwargs)

        # If in Carribbean or Tropics, dash a 50kt contour, too.
        if loc in ['Carrib','Tropics']:
            kwargs = {'colors':'blue',
                        'linestyles':'dashed',
                        'linewidth':1,
                        'levels':[50],
                        'transform':ccrs.PlateCarree()
                    }
            ax1.contour(lon, lat, masked_wspd, **kwargs)

        # Create legend label.
        legend2,_ = cspd.legend_elements()
        label2 = f'{lgd} Speed\n(every 25kt)'
    else:
        pass



    ######
    # Plot grid of magnitude of some other wind variable
    ######
    # What do you want to grid?
    if grid_fill in ['real']: use_grid=wspd
    elif grid_fill in ['geo']: use_grid=GEOspd
    elif grid_fill in ['ageo']: use_grid=aGEOspd
    elif grid_fill in ['ageo_along']: use_grid=aGEO_along_spddir
    elif grid_fill in ['ageo_perp']: use_grid=aGEO_perp_spddir
    elif grid_fill in ['real_div']: use_grid=wind_divergence
    elif grid_fill in ['geo_div']: use_grid=GEO_divergence
    elif grid_fill in ['ageo_div']: use_grid=aGEO_divergence

    # Adjust divergence from wind components to remove artifacts.
    elif grid_fill in ['ageo_along_div']: use_grid = adjust_div_values(lat,lon,smooth_hght,aGEO_along_divergence,aGEO_perp_divergence,grid_fill)
    elif grid_fill in ['ageo_perp_div']: use_grid = adjust_div_values(lat,lon,smooth_hght,aGEO_along_divergence,aGEO_perp_divergence,grid_fill)
    elif grid_fill is None: use_grid=None


    if use_grid is not None:
        ##
        # Create the colormap
        ##

        # convert wind units to kts.
        if use_grid.units in ['meter / second','knot']:
            use_grid = use_grid.to('kts')

        # What is the top and bottom extent (bounds) of the scale?
        if grid_fill in ['real','geo','ageo']:
            bounds = round(wind_1d_bound/10,0)*10
            interval = bounds*2
        elif grid_fill in ['ageo_along','ageo_perp']:
            bounds = round(wind_2d_bound/10,0)*10
            interval = bounds*2
        elif grid_fill[-4:]=='_div':
            colors_multiplier = 1000000  #10^6
            bounds = round(div_bound*colors_multiplier,0)/colors_multiplier
            interval = bounds*colors_multiplier*2
            print(f" --> bounds: {bounds}, interval: {interval}")

        # Get the base colormap
        if grid_fill in ['real','geo','ageo']:
            getcmap = cm.get_cmap('Blues',interval)
        elif grid_fill in ['ageo_along','ageo_perp']:
            getcmap = cm.get_cmap('RdBu',interval)
        elif grid_fill[-4:]=='_div':
            getcmap = cm.get_cmap('PuOr_r',interval)

        # Calculate how much white goes in the middle, assuming one pixel per speed unit.
        #   The np.ones([(min_grid*2)-1,_]) assures we have an odd number of rows to surround zero.
        #   The np.ones([_,4]) is me mimicking an rgba(1,1,1,1) value.
        #   The colors_multiplier converts tiny decimal values to integer values.
        if use_grid.units in ['meter / second','knot']:
            white = np.ones([(min_grid*2)-1,4])

        elif use_grid.units == '1 / second':
            white = np.ones([int((min_grid*colors_multiplier*2)-1),4])

        # Put the colorbar together.
        #   Take the base colormap, and vertically stack
        #   the top half, then white, then the bottom half.
        #   source: https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        if use_grid.units in ['meter / second','knot']:
            # How many colors on each side of zero?
            num_colors = int(bounds-min_grid)
            # Stack
            newcolors = np.vstack((getcmap(np.linspace(0, .45, num_colors)),
                                white,
                                getcmap(np.linspace(1-.45, 1, num_colors)) ))
            # Label on left of colorbar. 
            # ~ abbreviates units: knots -> kt
            # P shows units in "pretty print", H gives HTML code, L gives LATEX code.
            label = f'Speed ({use_grid.units:~P})'

        elif use_grid.units == '1 / second':
            num_colors = int(round(bounds-min_grid,5)*colors_multiplier)
            newcolors = np.vstack((getcmap(np.linspace(0, .4, num_colors)),
                                white,
                                getcmap(np.linspace(1-.4, 1, num_colors)) ))
            label = f'($10^{{{-5}}}$ {use_grid.units:~P})'

        # Use assembled colors to create and name a new colormap (cmp).
        newcmp = ListedColormap(newcolors, name='RedBlue')


        ##
        # After trimming the data and constructing the colorbar,
        # plot the grid of values.
        ##
        # Settings for plotting mesh and its colorbar.
        kwargs = {'vmin':-bounds,
                    'vmax':bounds,
                    'rasterized':True,
                    'cmap':newcmp,
                    'transform':ccrs.PlateCarree()
                }
        kwargs_cb = {'extend':'both',
                    'label':label,
                    'pad':0.03,
                    'shrink':0.75
                }
        if grid_fill in ['real','geo','ageo']:
            kwargs['vmin'] = min_grid
            kwargs['cmap'] = getcmap
            kwargs_cb['extend'] = 'max'
        elif grid_fill[-4:]=='_div':
            kwargs_cb['format'] = ticker.FuncFormatter(fmt)

        # Plot the data.
        gridspd = ax1.pcolormesh(lon, lat, use_grid, **kwargs)

        # Show the colorbar
        cb = fig1.colorbar(gridspd, ax=ax1, **kwargs_cb)

        # Label the colorbar.
        if bounds > 10: cb.set_label(label,size=15,labelpad=-62)
        else: cb.set_label(label,size=15,labelpad=-55)


        ##
        # Plot red areas where the divergence values have been adjusted.
        ##
        if red_on_map:
            # Calculate derivatives of height.
            dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
            hght_first = mpcalc.first_derivative(smooth_hght, delta=dy)

            # Flatten lats/lons
            lon_flat = lon.values.flatten()
            lat_flat = lat.values.flatten()
            hght_first = hght_first.flatten()

            # Find locations to highlight
            same_lon = []
            same_lat = []
            for i,row in enumerate(aGEO_along_divergence.flatten()):
                if -0.0001<hght_first[i]<0.0001:

                    # Identify lats/lons that qualify.
                    same_lon.append(lon_flat[i])
                    same_lat.append(lat_flat[i])

            print(f'Altered {len(same_lon)} of {lon.shape[0]*lon.shape[1]}: {len(same_lon)/(lon.shape[0]*lon.shape[1])}')

            # Draw red circles
            ax1.plot(same_lon,same_lat,'ro',alpha=0.25,ms=3,mew=0,transform=ccrs.Geodetic())


    ######
    # Plot Barbs or Quiver of some wind variable
    ######
    # Get the wind we want from settings.
    if plot_barbs in ['real']:
        u_wind = uwind[5:-5,5:-5]
        v_wind = vwind[5:-5,5:-5]
    elif plot_barbs in ['geo']:
        u_wind = uGEO[5:-5,5:-5]
        v_wind = vGEO[5:-5,5:-5]
    elif plot_barbs in ['ageo']:
        u_wind = uaGEO[5:-5,5:-5]
        v_wind = vaGEO[5:-5,5:-5]
    elif plot_barbs in ['ageo_along']:
        u_wind = aGEO_along_u[5:-5,5:-5]
        v_wind = aGEO_along_v[5:-5,5:-5]
    elif plot_barbs in ['ageo_perp']:
        u_wind = aGEO_perp_u[5:-5,5:-5]
        v_wind = aGEO_perp_v[5:-5,5:-5]
    elif plot_barbs==None:
        u_wind = None
        v_wind = None

    # Space the barbs or quiver arrows apart.
    if u_wind is not None:
        # Lat and Lon locations of barbs or quiver arrows.
        x = lon[0::spacing,0::spacing].values
        y = lat[0::spacing,0::spacing].values

        # U and V components of wind.
        u, v = mask_wind(u_wind,v_wind,wspd,min_barb,spacing)

        # Adjust units.
        u = u*units('m/s').to('kts')
        v = v*units('m/s').to('kts')

        # Plot wind barbs or arrows
        kwargs = {'label':'Wind barbs',
                    'transform':ccrs.PlateCarree()
                }
        if barb_quiver == 'barb':
            kwargs['length'] = 6
            GEObarb = ax1.barbs(x, y, u, v, **kwargs)
        elif barb_quiver == 'quiver':
            GEObarb = ax1.quiver(x, y, u, v, **kwargs)

        # Create legend label.
        GEObarb.set_label('Wind barbs')


    ######
    # With everything plotted, add titles, legend, etc.
    ######

    # Add legend
    kwargs = {'bbox_to_anchor':(1,0.99),
                'bbox_transform':ax1.transAxes,
                'title_fontsize':12,
                'fontsize':8,
                'ncol':3,
                'loc':4,
                'frameon':False
    }
    if plot_850_hghts:
        leg = ax1.legend([legend2[0],legend1[0],legend850[1]], [label2,label1,label850], **kwargs)
    else:
        kwargs['ncol'] = 2
        leg = ax1.legend([legend2[0],legend1[0]], [label2,label1], **kwargs)
    leg._legend_box.align = "right"


    # Add text around colorbar, providing extra meaning.
    if grid_fill == 'ageo_along':
        ylabel_text = "subgeostrophic\t\t\t\t\t\t\t    supergeostrophic  ".expandtabs()
        plt.gcf().text(0.78,0.5,ylabel_text,rotation='vertical',va='center',ha='right',snap=True)
    elif grid_fill == 'ageo_perp':
        ylabel_text = "exit region\t\t\t\t\t\t\t\t  entrance region   ".expandtabs()
        plt.gcf().text(0.78,0.5,ylabel_text,rotation='vertical',va='center',ha='right',snap=True)
    elif grid_fill[-4:]=='_div':
        ylabel_text = "convergence\t\t\t\t    divergence  ".expandtabs()
        plt.gcf().text(0.78,0.5,ylabel_text,size=15,rotation='vertical',va='center',ha='right',snap=True)


    # Add titles to the plot
    # pressure level, model name, model hour, date
    if not forecast:
        if name is not None:
            plt.title(f'{level}-hPa {name}\n', size=18, weight='bold')
            plt.title(f'\n0.25\N{DEGREE SIGN} GFS Analysis: {date:%H00 UTC, %B %d, %Y}', size=10, loc='left')
        else:
            plt.title(f'{level}-hPa\n', size=18, weight='bold')
            plt.title(f'\n0.25\N{DEGREE SIGN} GFS Analysis', loc='left')
    elif forecast:
        if name is not None:
            plt.title(f'{level}-hPa {name}\n', size=18, weight='bold')
            plt.title(f'\n0.25\N{DEGREE SIGN} GFS {fhr}-hour Forecast: {date+timedelta(hours=fhr):%H00 UTC, %B %d, %Y %z}', size=10, loc='left')
        else:
            plt.title(f'{level}-hPa\n', size=18, weight='bold')
            plt.title(f'\n0.25\N{DEGREE SIGN} GFS {fhr}-hour Forecast: {date+timedelta(hours=fhr):%H00 UTC, %B %d, %Y}', loc='left')


    # Add attribution label.
    text = '@jet_stream_bot'
    kwargs = {'weight':'bold',
                'bbox':dict(boxstyle="round",ec='white',fc="white",alpha=0.75),
                'va':'bottom',
                'snap':True,
                'transform':ax1.transAxes
            }
    ax1.text(0.006,0.01,text,**kwargs)


    ##################
    # Save the image #
    ##################
    savepath = f"{level}/{level}_{fhr}_{grid_fill}.png"
    plt.savefig(savepath,bbox_inches='tight')


    # Clear the axis.
    plt.clf()


    # Time to plot data
    time_elapsed = dt.now() - start_time
    tsec = round(time_elapsed.total_seconds(),2)
    print(f"--> Plotted data ({tsec:.2f} seconds)")

    return True




################################################
#                                              # 
# PRIMARY Functions                            #
# These are accessed by the main script below. #
#                                              # 
################################################

###########################
# Function 1:             #
# Get max colorbar values #
###########################
def get_max_colorbar(loc):
    """
    Function parameters:
        loc = location to plot. 'CONUS', 'Tropics', 'Carrib'
    """

    # Get values from the file and split them into a list.
    try:
        print(f"--> Get {loc}_max_colorbar_values.txt")
        with open(f'./{loc}_max_colorbar_values.txt','r') as file_of_maxes:
            lines = file_of_maxes.readlines()
            colorbar_maxes = lines[1].split(',')

    # If the file doesn't exist, create one with zeros as the values.
    # Report the zeros as the values, knowing they'll get updated later.
    except:
        print(f"--> Create new {loc}_max_colorbar_values.txt file.")
        header = 'one_sided_wind,two_sided_wind,divergence\n'
        new_max_values = '0,0,0'

        # Write file.
        with open('./{loc}_max_colorbar_values.txt','w') as file_of_maxes:
            file_of_maxes.write(header)
            file_of_maxes.write(new_max_values)

        colorbar_maxes = new_max_values.split(',')

    return colorbar_maxes



##############################################
# Function 2:                                #
# Script that controls making of images.     #
# This accesses a number of above functions. #
##############################################
def make_images(ds,date,fhr,levels,focus_level,colorbar_maxes,
                plots,num_passes,spacing,barb_quiver,
                plot_hghts,red_on_map,big_start_time):

    """
    Function parameters:
        ds = dataset (ds) to be plotted
        date = date of the data initialization time, for titles and such
        fhr = forecast hour(s) to plot
        levels = list of pressure levels to plot
        colorbar_maxes = how far should colorbar extend for each type of plot?
        plots = list of plots to make.
        num_passes = integer, number of smoothing passes to make
        spacing = integer, now many grids between barb or quiver
        barb_quiver = plot wind barbs or arrows. 'barb', 'quiver'
        big_start_time = datetime object of when the script started running
    """

    # For each pressure level...
    for level in levels:
        print(f"\n*** Forecast Hour {fhr} - {level}hPa - ({(dt.now() - big_start_time).total_seconds():.2f} seconds) ***\n")

        # Catch any problems before we begin.
        print("LOOKING FOR PROBLEMS")
        args = [plots,
                level,
                num_passes,
                spacing,
                barb_quiver,
                plot_hghts
                ]
        true_false = problems(*args)
        print("None found.")


        # Calculate all the wind and divergence variables to plot.
        print("\nMAKING CALCULATIONS")
        all_args = (ds,date,forecast,fhr)
        kwargs = {'level':level,
                'num_passes':num_passes,
                'loc':location
            }
        args_map, args_uv, args_spd, args_div = calculate_variables(*all_args,**kwargs)


        # Get the bounds we need for our colorbar.
        print("\nGETTING BOUNDS")
        restart_if_false, wind_1d_bound, wind_2d_bound, div_bound = get_bounds(colorbar_maxes, *args_spd, *args_div, location, replace)


        # If current data is beyond bound, and we're allowing the file to be updated,
        # Start making plots again, from the beginning, so plotting is uniform.
        if replace and not restart_if_false:

            # Reset the colorbar_maxes info.
            colorbar_maxes = get_max_colorbar(location)

            # Start the program from the beginning.
            make_images(ds,date,fhr,levels,focus_level,colorbar_maxes,plots,num_passes,spacing,barb_quiver,red_on_map,big_start_time)

            # Don't do steps below. They've already occurred.
            break

        else: pass


        # Group bounds together. 
        bounds_set = (wind_1d_bound, wind_2d_bound, div_bound)


        # Make scatter plot of the divergence components.
        print("\nMAKING SCATTER PLOT")
        diagnostic_scatter_plot(level,fhr,args_div,args_map,args_uv)


        # Plot the maps.
        print("\nMAKING MAPS")
        if level==focus_level:
            # Plot all map types for pressure level to focus on.
            for i,plot in enumerate(plots):
                print(f"\n--> Plotting #{i}: {plot['name']}")
                all_args = (args_map,args_uv,args_spd,args_div,date)
                kwargs = {'contour_wind':plot['contour_wind'],
                            'plot_barbs':plot['plot_barbs'],
                            'grid_fill':plot['grid_fill'],
                            'name':plot['name'],
                            'fhr':fhr,
                            'level':level,
                            'bounds_set':bounds_set,
                            'spacing':spacing,
                            'barb_quiver':barb_quiver,
                            'plot_hghts':plot_hghts,
                            'loc':location,
                            'red_on_map':red_on_map
                    }
                plot_the_map(*all_args,**kwargs)
        else:
            # Only plot what will be used for plots at other pressure levels.
            i = 0
            plot = plots[-1]
            #for i,plot in enumerate(plots[-1]):

            print(f"\n--> Plotting #{i}: {plot['name']}")
            all_args = (args_map,args_uv,args_spd,args_div,date)
            kwargs = {'contour_wind':plot['contour_wind'],
                        'plot_barbs':plot['plot_barbs'],
                        'grid_fill':plot['grid_fill'],
                        'name':plot['name'],
                        'fhr':fhr,
                        'level':level,
                        'bounds_set':bounds_set,
                        'spacing':spacing,
                        'barb_quiver':barb_quiver,
                        'plot_hghts':plot_hghts,
                        'loc':location,
                        'red_on_map':red_on_map
                }
            plot_the_map(*all_args,**kwargs)



########################################
# Function 3:                          #
# Make animations of the plotted maps. #
########################################
def make_animation(level,fhr_list):

    """
    Function parameters:
        level = pressure level in hectopascals or millibars.
    """

    # How long will it take to get the data?
    start_time = dt.now()

    # What files belong in the animation?
    # Master list contains lists of animations to make.
    # Note, this is actually one long list with four parts.
    files_list = []
    files_list_parts = ['real','ageo_along','ageo_perp','ageo_div']

    for a in files_list_parts:
        for b in fhr_list:
            files_list.append(f'{level}_{b}_{a}.png')

    files_list = [files_list]

    """
    files_list = [[f'{level}_0_real.png',f'{level}_3_real.png',f'{level}_6_real.png',f'{level}_9_real.png',f'{level}_12_real.png',f'{level}_15_real.png',f'{level}_18_real.png',f'{level}_21_real.png',f'{level}_24_real.png',f'{level}_27_real.png',f'{level}_30_real.png',f'{level}_33_real.png',f'{level}_36_real.png',f'{level}_39_real.png',f'{level}_42_real.png',f'{level}_45_real.png',f'{level}_48_real.png',
                f'{level}_0_ageo_along.png',f'{level}_3_ageo_along.png',f'{level}_6_ageo_along.png',f'{level}_9_ageo_along.png',f'{level}_12_ageo_along.png',f'{level}_15_ageo_along.png',f'{level}_18_ageo_along.png',f'{level}_21_ageo_along.png',f'{level}_24_ageo_along.png',f'{level}_27_ageo_along.png',f'{level}_30_ageo_along.png',f'{level}_33_ageo_along.png',f'{level}_36_ageo_along.png',f'{level}_39_ageo_along.png',f'{level}_42_ageo_along.png',f'{level}_45_ageo_along.png',f'{level}_48_ageo_along.png',
                f'{level}_0_ageo_perp.png',f'{level}_3_ageo_perp.png',f'{level}_6_ageo_perp.png',f'{level}_9_ageo_perp.png',f'{level}_12_ageo_perp.png',f'{level}_15_ageo_perp.png',f'{level}_18_ageo_perp.png',f'{level}_21_ageo_perp.png',f'{level}_24_ageo_perp.png',f'{level}_27_ageo_perp.png',f'{level}_30_ageo_perp.png',f'{level}_33_ageo_perp.png',f'{level}_36_ageo_perp.png',f'{level}_39_ageo_perp.png',f'{level}_42_ageo_perp.png',f'{level}_45_ageo_perp.png',f'{level}_48_ageo_perp.png',
                f'{level}_0_ageo_div.png',f'{level}_3_ageo_div.png',f'{level}_6_ageo_div.png',f'{level}_9_ageo_div.png',f'{level}_12_ageo_div.png',f'{level}_15_ageo_div.png',f'{level}_18_ageo_div.png',f'{level}_21_ageo_div.png',f'{level}_24_ageo_div.png',f'{level}_27_ageo_div.png',f'{level}_30_ageo_div.png',f'{level}_33_ageo_div.png',f'{level}_36_ageo_div.png',f'{level}_39_ageo_div.png',f'{level}_42_ageo_div.png',f'{level}_45_ageo_div.png',f'{level}_48_ageo_div.png']]
    """

    # What to name the files?
    filenames = [f'{level}_looking_ahead.gif']


    # Assemble each GIF.
    for i,files in enumerate(files_list):
        frames = []
        # Assemble each image into a list of frames for the GIF.
        # Resize the file so the GIF is ~10MB.
        for j,file in enumerate(files):
            new_frame = PIL.Image.open(f'{level}/{file}', mode='r')
            frames.append(new_frame.resize((682,454)))

            # Make the last image in each of the four parts last longer.
            if i==0 and j in [16,33,50,67]:
                for x in range(5): frames.append(new_frame.resize((682,454)))

        # Save GIF
        frames[0].save(
            f'{level}/{filenames[i]}',
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=200,               # time in milliseconds
            optimize=True,              # makes file smaller
            loop=0)                     # 0 = loops forever

        print(f'--> Made {filenames[i]} - {len(frames)} frames')

    # Time to plot data
    time_elapsed = dt.now() - start_time
    tsec = round(time_elapsed.total_seconds(),2)
    print(f"--> Made animations ({tsec:.2f} seconds)")

    return True



######################################
# Function 4:                        #
# Tweet a series of images and GIFs. #
######################################
def tweet_images(date,fhr,send_tweet):

    """
    Function parameters:
        date = the datetime format of the date.
        fhr = forecast hour that was plotted.
        send_tweet = True is send, False is don't send.
    """

    # Tweet the requested images or animations
    # 1) Real wind + GEO wind + divergence + aGEO wind
    # 2) Supergeostrophic wind + super-div
    # 3) 4-Quadrant wind + perp-div
    # 4) Div at 200, 250, 300, 400
    # 5) GIF looking ahead to 48-hr forecast.
    #       Real wind + Supergeostrophic + 4-Quadrant + divergence

    reply = False

    """
    # Test
    text = f"GFS Forecast\nData from {date:%A, %H UTC %-d %B %Y}\n"
    images = f"./200/200_looking_ahead.gif"
    tweet(text,images,send_tweet,reply)
    """


    # First tweet.
    text = f"GFS Forecast\nfor {date+timedelta(hours=fhr):%H UTC %-d %B}\n\n- forecasted wind\n- geostrophic wind\n- ageostrophic wind\n- divergence.\n\n{fhr}-hour forecast\nData from {date:%A, %H UTC %-d %B %Y}\n"
    images = [f"./{focus_level}/{focus_level}_{fhr}_real.png",
                f"./{focus_level}/{focus_level}_{fhr}_geo.png",
                f"./{focus_level}/{focus_level}_{fhr}_ageo.png",
                f"./{focus_level}/{focus_level}_{fhr}_ageo_div.png"]
    tweet(text,images,send_tweet,reply)

    time.sleep(5)


    # The rest of the tweets should be replies, forming a thread.
    reply = True


    # Second tweet
    text = f"Supergeostrophic wind and divergence\nAgeostrophic wind along the streamline.\n\n{fhr}-hour GFS {focus_level}-hPa forecast\nfor {date+timedelta(hours=fhr):%H UTC %-d %B}"
    images = [f"./{focus_level}/{focus_level}_{fhr}_ageo_along.png",
                f"./{focus_level}/{focus_level}_{fhr}_ageo_along_div.png"]
    tweet(text,images,send_tweet,reply)

    time.sleep(5)


    # Third tweet
    text = f"Four Quadrant Model wind and divergence\nAgeostrophic wind perpendicular to the streamline.\n\n{fhr}-hour GFS {focus_level}-hPa forecast\nfor {date+timedelta(hours=fhr):%H UTC %-d %B}"
    images = [f"./{focus_level}/{focus_level}_{fhr}_ageo_perp.png",
                f"./{focus_level}/{focus_level}_{fhr}_ageo_perp_div.png"]
    tweet(text,images,send_tweet,reply)

    time.sleep(5)


    # Fourth tweet
    text = f"Comparing altitudes\n\n{fhr}-hour GFS forecast\nfor {date+timedelta(hours=fhr):%H UTC %-d %B}"
    images = [f"./200/200_{fhr}_ageo_div.png",
                f"./250/250_{fhr}_ageo_div.png",
                f"./300/300_{fhr}_ageo_div.png",
                f"./400/400_{fhr}_ageo_div.png"]
    tweet(text,images,send_tweet,reply)

    time.sleep(5)


    # Fifth tweet
    text = f"Looking ahead 48 hours..."
    images = f"./{focus_level}/{focus_level}_looking_ahead.gif"
    tweet(text,images,send_tweet,reply)


    return True




################################
#                              # 
# MAIN script                  #
# This is where it all starts. #
#                              # 
################################



# How long does it take to run the script?
big_start_time = dt.now()



#############################
# Get date to retrieve data #
#############################
today = dt.utcnow()
if today.hour<6: hour = 0
elif today.hour<12: hour = 6
elif today.hour<18: hour = 12
else: hour = 18

# Assemble date, and move back by 6 hours for data availability
date = dt(today.year, today.month, today.day, hour)
date = date - timedelta(hours=6)



#####################################################
# Get GFS data for temperature and wind components. #
#####################################################
analysis_url = ('https://thredds.ucar.edu/thredds/dodsC/'
                'grib/NCEP/GFS/Global_0p25deg_ana/'
                'GFS_Global_0p25deg_ana')

forecast_url = ('https://thredds.ucar.edu/thredds/dodsC/'
                'grib/NCEP/GFS/Global_0p25deg/'
                'GFS_Global_0p25deg')

parse_fields = ['Geopotential_height_isobaric',
                'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric']


# Get Analysis Data
if not forecast:
    try:
        print(f"--> Getting analysis for {date:%Y-%m-%d %H UTC}")
        ds = xr.open_dataset(f'{analysis_url}_{date:%Y%m%d}_{date:%H}00.grib2')
        ds = ds.metpy.parse_cf(parse_fields)
        fhr_base = 0


    except:
        # Go back another synoptic time to find data.
        date = dt(today.year, today.month, today.day, hour) - timedelta(hours=6)
        fhr_base = 0

        print(f"--> OH NO! Getting analysis for {date:%Y-%m-%d %H UTC}")
        ds = xr.open_dataset(f'{analysis_url}_{date:%Y%m%d}_{date:%H}00.grib2')
        ds = ds.metpy.parse_cf(parse_fields)



# Get Forecast Data
elif forecast:
    try:
        print(f"--> Getting forecast from {date:%Y-%m-%d %H UTC}")
        ds = xr.open_dataset(f'{forecast_url}_{date:%Y%m%d}_{date:%H}00.grib2')
        ds = ds.metpy.parse_cf(parse_fields)

        # The 12th hour is the first one in the future when the data is available.
        fhr_base = 12

    except:
        # Go back another synoptic time to find data.
        date = dt(today.year, today.month, today.day, hour) - timedelta(hours=6)

        print(f"--> OH NO! Getting forecast from {date:%Y-%m-%d %H UTC}")
        ds = xr.open_dataset(f'{forecast_url}_{date:%Y%m%d}_{date:%H}00.grib2')
        ds = ds.metpy.parse_cf(parse_fields)

        # The 12th hour is the first one in the future when the data is available.
        fhr_base = 18



###########################
# Get max colorbar values #
###########################
colorbar_maxes = get_max_colorbar(location)



#########################################################
# Now that we have the date, data, and colorbar values, #
# let's do the calculations and create the plots.       #
#########################################################
for level in levels:
    if level==focus_level:
        use_levels = [level]
        # Calculate all forecast hours at pressure of focus level.
        for fhr in fhr_list:
            make_images(ds,date,fhr,use_levels,focus_level,colorbar_maxes,plots,num_passes,spacing,barb_quiver,plot_hghts,red_on_map,big_start_time)
    else:
        # Use only fhr_base for other pressure levels.
        use_levels = [level]
        make_images(ds,date,fhr_base,use_levels,focus_level,colorbar_maxes,plots,num_passes,spacing,barb_quiver,plot_hghts,red_on_map,big_start_time)



###########################################
# With the images made, make animated GIF #
###########################################
print(f"MAKING ANIMATIONS - ({(dt.now() - big_start_time).total_seconds():.2f} seconds)")
# Make animations from the maps you created.
for level in levels:
    if level==focus_level:
        true_false = make_animation(level,fhr_list)



###############################################
# With all the plots made, let's tweet stuff. #
###############################################
print(f"TWEETING - ({(dt.now() - big_start_time).total_seconds():.2f} seconds)")
tweet_images(date,fhr_base,send_tweet)



################################
# Total time to run the script #
################################
big_time_elapsed = dt.now() - big_start_time
tsec = round(big_time_elapsed.total_seconds(),2)
print(f"\n--> Done ({tsec:.2f} seconds)")



#######
# END #
#######
