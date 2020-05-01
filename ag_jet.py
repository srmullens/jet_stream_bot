
########################################################################
#                                                                      #
# This code takes GFS data, identifies the jet stream and jet streaks, #
# and plots the ageostrophic components of wind along and across       #
# the calculated geostrophic wind. Divergence of wind, geostrophic     #
# wind, and components of ageostrophic wind is also calculated.        #
#                                                                      #
# Code by Stephen Mullens. April 2020.                                 #
#                                                                      #
########################################################################

from datetime import datetime as dt, timedelta
import os

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


# UPDATES TO DO
#   Create animated GIFs.
#   Need to create standard text for tweets.


######################################################################
#
# Initial Functions
#

# Raise warnings if something won't work.
def problems(plots,level,num_passes,spacing,barb_quiver,plot_hghts):
#def problems(contour_wind,plot_barbs,grid_fill,level,num_passes,spacing,barb_quiver,plot_hghts):
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

    return True


# Format the labels on the colorbar.
def fmt(x, pos):
    # Make sure it's 10.0, not 10.0000000001
    x = round(x,5)

    # if 2x10^-4, a=2, b=-4
    a,b = f'{x:.1e}'.split('e')

    # I want all numbers to be 10^(-5). Adjust as needed.
    exp = int(b)+5
    a = float(a)*(10**exp)
    b = int(b)-exp

    # Create the final string.
    #sci = f'${a:.0f}\N{MULTIPLICATION SIGN}10^{{{b}}}$'
    sci = f'{a:.0f}'

    # If zero, return 0. Otherwise, return the string.
    if a==0.0 and b==0.0: return '0'
    else: return sci



# Find the maximum element everywhere in the grid of values.
# Given multiple wind values or multiple divergence values.
def max_arrays(*args):
    list_of_arg_maxes = []
    for arg in args:
        if arg.units in ['meter / second']: arg=arg.to('kts')
        arg = np.absolute(arg.magnitude)
        arg = np.nanmax(arg)
        list_of_arg_maxes.append(arg)

    max_value = np.nanmax(list_of_arg_maxes)

    return max_value



# Get the maximum values for the colorbars from a file.
# Make sure the current values don't exceed those from the file.
# If they do, replace the file.
def get_bounds(colorbar_maxes, wspd, GEOspd, aGEOspd, aGEO_along_spddir, aGEO_perp_spddir, wind_divergence, GEO_divergence, aGEO_divergence, aGEO_along_divergence, aGEO_perp_divergence):

    # Compare to max values. See if any of them need to be replaced.
    restart_if_false = True

    # If you find a max value bigger than what's been recorded, restart the loop of plots.
    if colorbar_maxes is not None:
        #colorbar_maxes = colorbar_maxes.split(',')
        replace1 = float(colorbar_maxes[0])
        replace2 = float(colorbar_maxes[1])
        replace3 = float(colorbar_maxes[2])
        print(f"--> old: {replace1:.2f}, {replace2:.2f}, {replace3:.2e}")
        this1 = max_arrays(wspd,GEOspd,aGEOspd)
        this2 = max_arrays(aGEO_along_spddir,aGEO_perp_spddir)
        this3 = max_arrays(wind_divergence,GEO_divergence,aGEO_divergence,aGEO_along_divergence,aGEO_perp_divergence)
        max1 = max2 = max3 = False

        print(f"--> Max: {this1:.2f}, {this2:.2f}, {this3:.2e}")

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
        if np.any([max1,max2,max3]):
            os.rename('./max_colorbar_values.txt','./max_colorbar_values_old.txt')
            header = 'one_sided_wind,two_sided_wind,divergence\n'
            replace = f"{replace1},{replace2},{replace3}"

            file_of_maxes = open('./max_colorbar_values.txt','w')
            file_of_maxes.write(header)
            file_of_maxes.write(replace)
            file_of_maxes.close()

    return restart_if_false,replace1,replace2,replace3



#####################
# Primary functions #
#####################
def calculate_variables(data=None, date=None, level=200, num_passes=40):

    # How long will it take to get the data?
    start_time = dt.now()

    """
    Function parameters:
        date = datetime object
        level = pressure level for the plot in hectopascals or millibars.
                    Typically in the jet stream.
        num_passes = controls how smooth the plot is.
                    40 passes means data 10 degrees away has an influence here.
    """


    ######################################################################
    # Observation Data
    # ----------------
    #
    # Set a date and time for upper-air observations (should only be 00 or 12
    # UTC for the hour).
    #
    # Request all data from Iowa State using the Siphon package. The result is
    # a pandas DataFrame containing all of the sounding data from all
    # available stations.
    #

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

    elif isinstance(date, dt):
        print(f"--> Using given date: {date:%Y-%m-%d %H UTC}")
        date = date

    else:
        warn=f'Date is type {type(date)}. Must be type datetime.datetime.'
        raise TypeError(warn)



    ######################################################################
    # STEP 1:
    # Gridded Data
    # ------------
    #
    # Obtain GFS gridded output for contour plotting. Specifically,
    # geopotential height and temperature data for the given level and subset
    # for over North America. Data are smoothed for aesthetic reasons.
    #

    if data is None:
        print("--> Data not given. Getting data")
        # Get GFS data and subset to North America for Geopotential Height and Temperature
        ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg_ana/'
                             f'GFS_Global_0p25deg_ana_{date:%Y%m%d}_{date:%H}00.grib2').metpy.parse_cf(['Geopotential_height_isobaric','u-component_of_wind_isobaric','v-component_of_wind_isobaric'])
    else:
        print("--> Using provided data")
        ds = data



    #
    # Get and Smooth Basic Parameters
    # 

    # Parameters of the location we're looking for.
    location = {'vertical':level*units.hPa,
                'time':date,
                'lat':slice(70, 15),
                'lon':slice(360-145, 360-50)
    }

    # Geopotential height and smooth
    hght = ds['Geopotential_height_isobaric'].metpy.loc[location]
    smooth_hght = mpcalc.smooth_n_point(hght, 9, num_passes)
    use_hght = smooth_hght


    # U and V wind components
    uwind = ds['u-component_of_wind_isobaric'].metpy.loc[location]
    vwind = ds['v-component_of_wind_isobaric'].metpy.loc[location]
    uwind = mpcalc.smooth_n_point(uwind, 9, num_passes)
    vwind = mpcalc.smooth_n_point(vwind, 9, num_passes)



    ######################################################################
    # STEP 2:
    # Manipulate the wind data
    # ------------
    #
    # Calculate the Coriolis (f), Geostrophic (GEO), and Ageostrophic (aGEO) wind
    # Calculate the divergence of the real, GEO, and aGEO winds.

    # Get the mapping crs from the data.
    data_crs = hght.metpy.cartopy_crs

    # Get the x, y, lat, and lon lists from the data.
    x, y = hght.metpy.coordinates('x','y')
    lat, lon = xr.broadcast(y, x)

    # Use lats and lons to get change in x and y in meters.
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat, initstring=data_crs.proj4_init)

    # Let MetPy calculate the Coriolis parameter.
    f = mpcalc.coriolis_parameter(lat)

    # Let MetPy calculate the geostrophic and ageostrophic wind components
    uGEO, vGEO = mpcalc.geostrophic_wind(use_hght, f, dx, dy, dim_order='yx')
    uaGEO, vaGEO = ageostrophic_wind(use_hght, uwind, vwind, f, dx, dy, dim_order='yx')



    ######################################################################
    # STEP 3: Use GEO and aGEO to calculate aGEO components
    #
    # Ccalculate GEO unit vectors,
    # direction of GEO wind and perpendicular to GEO wind,
    # aGEO wind components along and perpendicular to GEO wind,
    # the signed direction of aGEO components relative to GEO wind,
    # and the divergence from aGEO components relative to GEO wind.
    #

    # Let MetPy calculate actual, geostrophic, and ageostrophic wind speeds.
    wspd = mpcalc.wind_speed(uwind,vwind)
    GEOspd = mpcalc.wind_speed(uGEO,vGEO)
    aGEOspd = mpcalc.wind_speed(uaGEO,vaGEO)

    # Calculate u and v unit vector components of GEO.
    uGEO_unit = uGEO / GEOspd
    vGEO_unit = vGEO / GEOspd


    # Calculate the magnitude of aGEO wind along and perpendicular
    # to the GEO unit vector.
    #   unit_vector_angle = angle of GEO unit vector counter-clockwise from eastward direction.
    #   perp_vector_angle = 90 degrees counter-clockwise from unit_vector_angle.
    #
    #   aGEO_along_GEO = magnitude of aGEO wind directly with/against GEO wind.
    #   aGEO_perp_GEO = magnitude of aGEO wind perpendicular to GEO wind.
    unit_vector_angle = np.arctan2(vGEO_unit,uGEO_unit)
    perp_vector_angle = unit_vector_angle + (np.pi/2)*units('radian')

    aGEO_along_GEO = uaGEO*np.cos(unit_vector_angle) + vaGEO*np.sin(unit_vector_angle)
    aGEO_perp_GEO = -uaGEO*np.sin(unit_vector_angle) + vaGEO*np.cos(unit_vector_angle)


    # Preserve where the aGEO components are postiive or negative.
    #   Divide the number by its magnitude. Thus, if value >=0.0: =1; else: =-1

    #   Upstream is negative. Downstream is positive.
    direction_along_GEO = aGEO_along_GEO / np.absolute(aGEO_along_GEO)
    #   Left is positive. Right is negative.
    direction_perp_GEO = aGEO_perp_GEO / np.absolute(aGEO_perp_GEO)


    # Convert those magnitudes back to normal x/y coordinates.
    #   aGEO_along_u = E-W component of aGEO wind in direction of GEO wind
    #   aGEO_along_v = N-S component of aGEO wind in direction of GEO wind
    #
    #   aGEO_perp_u = E-W component of aGEO wind perpendicular to GEO wind
    #   aGEO_perp_v = N-S component of aGEO wind perpendicular to GEO wind

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



    ######################################################################
    # STEP 4: Calculate the divergence of all the parameters.
    #

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



    ######################################################################
    # Trim the data.
    # Output the final calculated numbers.
    #

    lon = lon[5:-5,5:-5]
    lat = lat[5:-5,5:-5]

    smooth_hght = smooth_hght[5:-5,5:-5]

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


    args_map = (lat,lon,smooth_hght)
    args_uv = (uwind,vwind, uGEO,vGEO, uaGEO,vaGEO, aGEO_along_u,aGEO_along_v, aGEO_perp_u,aGEO_perp_v)
    args_spd = (wspd, GEOspd, aGEOspd, aGEO_along_spddir, aGEO_perp_spddir)
    args_div = (wind_divergence, GEO_divergence, aGEO_divergence, aGEO_along_divergence, aGEO_perp_divergence)



    # Time to get data
    time_elapsed = dt.now() - start_time
    tsec = round(time_elapsed.total_seconds(),2)
    print(f"--> Got data ({tsec:.2f} seconds)")



    return args_map, args_uv, args_spd, args_div
### End calculate_variables()





######################################################################
# STEP 5: Plot the map
# ------------------------
#
# Plot the analyzed contours on a Lambert Conformal map.
#
def plot_the_map(args_map,args_uv,args_spd,args_div,date,
            contour_wind='real', plot_barbs='real', grid_fill='real', name=None,
            level=200,
            bounds_set=None,
            spacing=10, barb_quiver='barb', plot_hghts=True, extent=None):

    # How long will it take to get the data?
    start_time = dt.now()

    """
    Function parameters:
        args_map = lat, lon, height data for plotting.
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
        level = pressure level for the plot in hectopascals or millibars.
                    Typically in the jet stream. Used in title of the plot.
        bounds_set = set of extreme values to use when plotting grid colorbars.
        spacing = plots barbs every x grid points. Default is 10.
        barb_quiver = plots either barbs (default) or quivers (aka arrows).
        plot_hghts = Include the black height contours or not.
        extent = [W, E, S, N] list of latitudes where plot is located.
    """

    # Unpack the data to plot
    lat,lon,smooth_hght = args_map
    uwind,vwind, uGEO,vGEO, uaGEO,vaGEO, aGEO_along_u,aGEO_along_v, aGEO_perp_u,aGEO_perp_v = args_uv
    wspd, GEOspd, aGEOspd, aGEO_along_spddir, aGEO_perp_spddir = args_spd
    wind_divergence, GEO_divergence, aGEO_divergence, aGEO_along_divergence, aGEO_perp_divergence = args_div
    wind_1d_bound, wind_2d_bound, div_bound = bounds_set


    ######################################################################
    # Set minimum and maximum values for plotting contours, grids, barbs.
    # -------------------------
    # Because of above settings...
    #    ...set these settings.
    #

    # What minimum value wind speed do you want to contour? (knots)
    if contour_wind in ['real','geo']:
        min_contour = 75
    elif contour_wind in ['ageo']:
        min_contour = 30

    # What minimum value do you want barbs or quivers plotted for? (knots)
    if plot_barbs in ['real','geo']:
        min_barb = 75
    elif plot_barbs in ['ageo']:
        min_barb = 15
    elif plot_barbs in ['ageo_along','ageo_perp']:
        min_barb = 10

    # What minimum value do you want the grid fill to be colored in?
    if grid_fill in ['real','geo']:
        min_grid = 50
    elif grid_fill in ['ageo']:
        min_grid = 15
    elif grid_fill in ['ageo_along','ageo_perp']:
        min_grid = 10
    elif grid_fill in ['real_div','geo_div','ageo_div','ageo_along_div','ageo_perp_div']:
        min_grid = 0.00002


    ######################################################################
    # Subset Observational Data
    # -------------------------
    # The contour interval is set based on the level chosen.
    # If you want to label the contours, the label format is also given.
    #

    if (level == 925) | (level == 850) | (level == 700):
        cint = 30
        def hght_format(v): return format(v, '.0f')[1:]
    elif level == 500:
        cint = 60
        def hght_format(v): return format(v, '.0f')[:3]
    elif level == 300:
        cint = 120
        def hght_format(v): return format(v, '.0f')[:3]
    elif level < 300:
        cint = 120
        def hght_format(v): return format(v, '.0f')[1:4]



    # Set up map coordinate reference system
    mapcrs = ccrs.LambertConformal(
        central_latitude=45, central_longitude=-100, standard_parallels=(30, 60))

    # Start figure
    fig1 = plt.figure(1, figsize=(17, 15))
    #fig1 = plt.figure(1, figsize=(16, 9), dpi=300)
    ax1 = plt.subplot(111, projection=mapcrs)


    # Set map extent
    if extent is None:
        ax1.set_extent([-125, -70, 20, 55])
    elif isinstance(extent, list) and len(extent)==4:
        ax1.set_extent(extent)
    else:
        warn = f'extent must be a list of length four: [W, E, S, N]\nYou gave {type(extent)}, length {len(extent)}, {extent}.'
        raise RuntimeError(warn)


    # Add map features for geographic reference
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='grey')
    ax1.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white')
    ax1.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='grey')


    ######
    # Plot Solid Contours of Geopotential Height
    ######
    if plot_hghts:
        kwargs = {'colors':'black',
                'levels':range(0, 20000, cint),
                'transform':ccrs.PlateCarree()
                }
        cs = ax1.contour(lon, lat, smooth_hght, **kwargs)
        legend1,_ = cs.legend_elements()
        label1 = f'Heights (every {cint}m)'
    else: pass



    ######
    # Plot contour of wind speed
    ######
    if contour_wind in ['real']:
        use_wind = wspd
    elif contour_wind in ['geo']:
        use_wind = GEOspd
    elif contour_wind in ['ageo']:
        use_wind = aGEOspd
    elif contour_wind is None:
        use_wind==None

    if use_wind is not None:
        # Mask the real wind speed.
        masked_wspd = []
        for i,w in enumerate(use_wind):
            w = w.to('kts')
            masked_w = np.ma.masked_where(w < 50*units('kts'), w)
            masked_wspd.append(masked_w)
        masked_wspd = masked_wspd*units('kts')

        # Contour
        kwargs = {'colors':'blue',
                    'levels':range(min_contour,int(np.nanmax(masked_wspd.magnitude)+1),25),
                    'transform':ccrs.PlateCarree()
                }
        cspd = ax1.contour(lon, lat, masked_wspd, **kwargs)
        legend2,_ = cspd.legend_elements()
        label2 = 'Isotachs (50kt, 75kt, 100kt)'
    else:
        pass



    ######
    # Plot grid of magnitude of some other wind variable
    ######
    # What do you want to grid?
    # What minimum value do you want the grid fill to be colored in?
    if grid_fill in ['real']: use_grid=wspd
    elif grid_fill in ['geo']: use_grid=GEOspd
    elif grid_fill in ['ageo']: use_grid=aGEOspd
    elif grid_fill in ['ageo_along']: use_grid=aGEO_along_spddir
    elif grid_fill in ['ageo_perp']: use_grid=aGEO_perp_spddir
    elif grid_fill in ['real_div']: use_grid=wind_divergence
    elif grid_fill in ['geo_div']: use_grid=GEO_divergence
    elif grid_fill in ['ageo_div']: use_grid=aGEO_divergence
    elif grid_fill in ['ageo_along_div']: use_grid=aGEO_along_divergence
    elif grid_fill in ['ageo_perp_div']: use_grid=aGEO_perp_divergence
    elif grid_fill is None: use_grid=None


    if use_grid is not None:
        ##
        # Create the color scale
        ##
        # convert units to kts.
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


        # Get the colormap
        if grid_fill in ['real','geo','ageo']:
            getcmap = cm.get_cmap('Blues',interval)
        else:
            getcmap = cm.get_cmap('RdBu',interval)

        # Calculate how much white is in the middle, assuming one pixel per speed unit.
        #   The [(min_grid*2)-1,_] assures we have an odd number of rows to surround zero.
        #   [_,4] columns is me mimicking an rgba(1,1,1,1) value.
        if use_grid.units in ['meter / second','knot']:
            white = np.ones([(min_grid*2)-1,4])

        elif use_grid.units == '1 / second':
            white = np.ones([int((min_grid*colors_multiplier*2)-1),4])

        # Put the colorbar together.
        #   We're using the colormap we got, and are vertically stacking 
        #   the top half, then white, then bottom half.
        #       source: https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        if use_grid.units in ['meter / second','knot']:
            num_colors = int(bounds-min_grid)
            newcolors = np.vstack((getcmap(np.linspace(0, .45, num_colors)),
                                white,
                                getcmap(np.linspace(1-.45, 1, num_colors)) ))
            label = f'Wind speed ({use_grid.units:~P})'

        elif use_grid.units == '1 / second':
            num_colors = int(round(bounds-min_grid,5)*colors_multiplier)
            newcolors = np.vstack((getcmap(np.linspace(0, .4, num_colors)),
                                white,
                                getcmap(np.linspace(1-.4, 1, num_colors)) ))
            label = f'Divergence ($10^{{{-5}}}$ {use_grid.units:~P})'

        newcmp = ListedColormap(newcolors, name='RedBlue')


        ##
        # After trimming the data and constructing the colorbar,
        # plot the grid of values.
        ##
        kwargs = {'vmin':-bounds,
                    'vmax':bounds,
                    'rasterized':True,
                    'cmap':newcmp,
                    'transform':ccrs.PlateCarree()
                }
        kwargs_cb = {'extend':'both',
                    'label':label,
                    'pad':0.03,
                    #'labelpad':-1,
                    'shrink':0.45
                }
        if grid_fill in ['real','geo','ageo']:
            kwargs_cb['extend'] = 'max'
            kwargs['vmin'] = min_grid
            kwargs['cmap'] = getcmap
        elif grid_fill[-4:]=='_div':
            kwargs_cb['format'] = ticker.FuncFormatter(fmt)
        gridspd = ax1.pcolormesh(lon, lat, use_grid, **kwargs)
        cb = fig1.colorbar(gridspd, ax=ax1, **kwargs_cb)
        if grid_fill[-4:]=='_div': cb.set_label(label,size=15,labelpad=-61)
        else: cb.set_label(label,size=15,labelpad=-61)


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


    def mask_wind(U,V,min_speed,spacing):
        # create masked wind speed
        w = mpcalc.wind_speed(U,V)
        w = w.to('kts')
        mask = []
        for i,values in enumerate(U):
            create_mask = np.ma.masked_where(w[i] < min_speed*units('kts'), wspd[i])
            mask.append(np.ma.getmask(create_mask))

        # apply the mask to u and v components
        masked_u = np.ma.masked_array(U,mask=mask)
        masked_v = np.ma.masked_array(V,mask=mask)

        # apply spacing
        masked_u = masked_u[0::spacing,0::spacing]
        masked_v = masked_v[0::spacing,0::spacing]

        return masked_u, masked_v

    if u_wind is not None:
        x = lon[0::spacing,0::spacing].values
        y = lat[0::spacing,0::spacing].values
        u, v = mask_wind(u_wind,v_wind,min_barb,spacing)
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

        GEObarb.set_label('Wind barbs')

    ######
    # With everything plotted, add titles, legend, etc.
    ######

    # Add legend
    kwargs = {'bbox_to_anchor':(1,0.99),
                'bbox_transform':ax1.transAxes,
                #'title':f'0.25\N{DEGREE SIGN} GFS Analysis',
                'title_fontsize':12,
                'ncol':2,
                'loc':4,
                'frameon':False
    }
    leg = ax1.legend([legend1[0],legend2[0]], [label1,label2], **kwargs)
    leg._legend_box.align = "right"

    # Add titles
    if name is not None:
        plt.title(f'{level}-hPa {name}\n', size=18, weight='bold')
        plt.title(f'\n0.25\N{DEGREE SIGN} GFS Analysis: {date:%I00 UTC, %B %d, %Y}', size=10, loc='left')
    else:
        ax1.title(f'{level}-hPa\n', size=18, weight='bold')
        plt.title(f'\n0.25\N{DEGREE SIGN} GFS Analysis', loc='left')

    # Save the image
    #savepath = f"{date:%Y%m%d%H}_{level}_{grid_fill}.png"
    savepath = f"{level}_{grid_fill}.png"
    plt.savefig(savepath,bbox_inches='tight')


    # Clear the axis.
    plt.clf()


    # Time to plot data
    time_elapsed = dt.now() - start_time
    tsec = round(time_elapsed.total_seconds(),2)
    print(f"--> Plotted data ({tsec:.2f} seconds)")

    return restart_if_false
### End plot_the_map()




######################################################################
# STEP 5: Plot the map
# ------------------------
#
# Plot the analyzed contours on a Lambert Conformal map.
#
def make_animation(level):

    # How long will it take to get the data?
    start_time = dt.now()

    # Make animation
    files_list = [[f'{level}_real.png',f'{level}_geo.png'],
            [f'{level}_real.png',f'{level}_ageo_along.png',f'{level}_ageo_perp.png'],
            [f'{level}_ageo_perp.png',f'{level}_ageo_perp_div.png'],
            [f'{level}_ageo_along.png',f'{level}_ageo_along_div.png'],
            [f'{level}_real.png',f'{level}_geo.png',f'{level}_ageo.png',f'{level}_ageo_along.png',f'{level}_ageo_along_div.png',f'{level}_ageo_perp.png',f'{level}_ageo_perp_div.png',f'{level}_ageo_div.png'],
            [f'{level}_real.png',f'{level}_ageo_div.png']
    ]
    filenames = [f'{level}_real_vs_geo.gif',f'{level}_ageo_components.gif',f'{level}_ageo_perp.gif',f'{level}_ageo_along.gif',f'{level}_div_components.gif',f'{level}_divergence.gif']

    for i,files in enumerate(files_list):
        frames = []
        print()
        for j,file in enumerate(files):
            print('Appending file', file)
            new_frame = PIL.Image.open(file, mode='r')
            frames.append(new_frame)
            if j>0: frames.append(new_frame)

        """
        # Make last frame last longer
        for i in range(10):
            print(file)
            frames.append(new_frame)
        """

        print(len(frames))

        # Save gif
        frames[0].save(
            filenames[i],
            format='GIF',
            append_images=frames,
            save_all=True,
            duration=400*len(files),
            optimize=True,
            loop=0)  # forever

    # Time to plot data
    time_elapsed = dt.now() - start_time
    tsec = round(time_elapsed.total_seconds(),2)
    print(f"--> Made animations ({tsec:.2f} seconds)")

    return True





######################################################################

# How long will it take to get the data?
big_start_time = dt.now()

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

#plots = [{'name':'Wind','contour_wind':'real','plot_barbs':'real','grid_fill':'real'}]
#plots = [{'name':'Divergence and Wind','contour_wind':'real','plot_barbs':'real','grid_fill':'ageo_div'}]


plots = [{'name':'Wind','contour_wind':'real','plot_barbs':'real','grid_fill':'real'},
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
levels = [200]

# Do you want height contours?
plot_hghts = True

# smoothing: number of times (passes) data is smoothed
num_passes = 40

# Plot wind barbs or quivers every X grid places.
spacing = 10

# Barbs or Quiver?
barb_quiver = 'barb'

# Where do you want the map to be plotted?
extent = [-125, -70, 20, 55]



###########################
# Get max colorbar values #
###########################
try:
    print("--> Get max_colorbar_values.txt")
    with open('./max_colorbar_values.txt','r') as file_of_maxes:
        lines = file_of_maxes.readlines()
        colorbar_maxes = lines[1].split(',')

except:
    print("--> Create new max_colorbar_values.txt file.")
    file_of_maxes = open('./max_colorbar_values.txt','w')
    header = 'one_sided_wind,two_sided_wind,divergence\n'
    new_max_values = '0,0,0'
    # Write file.
    file_of_maxes.write(header)
    file_of_maxes.write(new_max_values)
    file_of_maxes.close()
    colorbar_maxes = new_max_values


#############################
# Set date for desired data #
#############################
today = dt.utcnow()
if today.hour<6: hour = 0
elif today.hour<12: hour = 6
elif today.hour<18: hour = 12
else: hour = 18


############################################################
# Get GFS data for Geopotential Height and Wind components #
############################################################
try:
    date = dt(today.year, today.month, today.day, hour)
    print(f"--> Getting data for {date:%Y-%m-%d %H UTC}")
    ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg_ana/'
                         f'GFS_Global_0p25deg_ana_{date:%Y%m%d}_{date:%H}00.grib2').metpy.parse_cf(['Geopotential_height_isobaric','u-component_of_wind_isobaric','v-component_of_wind_isobaric'])

except:
    # Go back one synoptic time to ensure data availability
    date = dt(today.year, today.month, today.day, hour) - timedelta(hours=6)
    print(f"--> OH NO! Getting data for {date:%Y-%m-%d %H UTC}")

    ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg_ana/'
                         f'GFS_Global_0p25deg_ana_{date:%Y%m%d}_{date:%H}00.grib2').metpy.parse_cf(['Geopotential_height_isobaric','u-component_of_wind_isobaric','v-component_of_wind_isobaric'])



###############
# Go to work! #
###############

for level in levels:
    print(f"\n*** {level}hPa ***\n")
    """
    print("LOOKING FOR PROBLEMS")
    # Catch any problems before we begin.
    args = [plots,
            level,
            num_passes,
            spacing,
            barb_quiver,
            plot_hghts
            ]
    true_false = problems(*args)

    print("MAKING CALCULATIONS")
    # Calculate all the wind and divergence variables to plot.
    all_args = (ds,date)
    kwargs = {'level':level,
            'num_passes':num_passes}

    args_map, args_uv, args_spd, args_div = calculate_variables(*all_args,**kwargs)

    print("GETTING BOUNDS")
    # Get the bounds we need for our colorbar.
    restart_if_false, wind_1d_bound, wind_2d_bound, div_bound = get_bounds(colorbar_maxes, *args_spd, *args_div)
    bounds_set = (wind_1d_bound, wind_2d_bound, div_bound)

    print("MAKING MAPS")
    # Plot the maps.
    for i,plot in enumerate(plots):
        print(f"\n--> Plotting #{i}: {plot['name']}")
        all_args = (args_map,args_uv,args_spd,args_div,date)
        kwargs = {'contour_wind':plot['contour_wind'],
                    'plot_barbs':plot['plot_barbs'],
                    'grid_fill':plot['grid_fill'],
                    'name':plot['name'],
                    'level':level,
                    'bounds_set':bounds_set,
                    'spacing':spacing,
                    'barb_quiver':barb_quiver,
                    'plot_hghts':plot_hghts,
                    'extent':extent
            }
        plot_the_map(*all_args,**kwargs)
    """
    print("MAKING ANIMATIONS")
    # Make animations from the maps you created.
    true_false = make_animation(level)


# Time to plot data
big_time_elapsed = dt.now() - big_start_time
tsec = round(big_time_elapsed.total_seconds(),2)
print(f"\n--> Done ({tsec:.2f} seconds)")

# END
