import logging
import glob
import os
from ConfigParser import ConfigParser
import glob
import numpy as np
import numpy.ma as ma
from mpop import CONFIG_PATH

import pyresample
import datetime
from time import gmtime

from os.path import dirname, exists
from os import makedirs

import logging
#LOG = logging(__name__)



def load(satscene, *args, **kwargs):
    """Loads the *channels* into the satellite *scene*.
    """

    # Dataset information
    #
    # Read config file content
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    print "... read lightning config from ", os.path.join(CONFIG_PATH, satscene.fullname + ".cfg")

    values = {"orbit": satscene.orbit,
              "satname": satscene.satname,
              "number": satscene.number,
              "instrument": satscene.instrument_name,
              "satellite": satscene.fullname
              }

    # print values
    if "area" in kwargs:
        projectionName = kwargs["area"]
    else:
        projectionName = conf.get("thx-level2", "projection")
        #projectionName = "ccs4"
    print "... project lightning data to ", projectionName
    #scale = conf.get("radar-1", "scale")
    #print '... read scale from: ', scale

    gmt = gmtime()
    now = datetime.datetime(gmt.tm_year, gmt.tm_mon, gmt.tm_mday, gmt.tm_hour, gmt.tm_min, 0) 
    dtime = now - satscene.time_slot
    print "... desired time slot: "+ str(satscene.time_slot)+ ", current time: "+ str(now)
    print "    time difference: "+ str(dtime)
    NEAR_REAL_TIME = False

    archive_time = datetime.timedelta(days=5) # last 5 days are in the archive
    if dtime < archive_time:
        print "    use the near real time data "
        NEAR_REAL_TIME=True
        filename = os.path.join(
            satscene.time_slot.strftime(conf.get("thx-level2", "nearrealtime_dir")),
            satscene.time_slot.strftime(conf.get("thx-level2", "nrt_filename", raw=True)) % values)
    else: 
        data_dir = satscene.time_slot.strftime(conf.get("thx-level2", "archive_dir"))
        print "    use the all time archive "
        filename = os.path.join(
            satscene.time_slot.strftime(conf.get("thx-level2", "archive_dir")),
            satscene.time_slot.strftime(conf.get("thx-level2", "archive_filename", raw=True)) % values)
        
    ## my old solution 
    #filename = os.path.join(
    #    satscene.time_slot.strftime(conf.get("thx-level2", "dir")),
    #    satscene.time_slot.strftime(conf.get("thx-level2", "filename", raw=True)) % values)
    #NEAR_REAL_TIME=True

    # get spatial and temporal resolution
    if "dt" in kwargs:
        dt = float(kwargs["dt"])
        print "... dt specified by function argument: ", dt
        print "!!! Attention, argument dt is only implemented for archived data !!!"
    else:
        dt = float(conf.get("thx-level2", "dt"))
        print "... dt specified by configuration file:", dt
    print "... read the observation of the last ", dt, " min"

    # Load data from txt file 
    print "... read data from", str(filename), os.path.dirname(filename)

    if not os.path.isdir(os.path.dirname(filename)):
        print '*** ERROR, lightning input directory '+str(os.path.dirname(filename))+' cannot be accessed'
        print '    please check the config file'
        quit()
    elif len(glob.glob(str(filename))) == 0:
        "*** WARNING, no file "+str(filename)+" found!"
        filename=""
    elif len(glob.glob(str(filename))) > 1:
        "*** WARNING, more than one file "+str(filename)+" found!"
        filename = glob.glob(str(filename))[0]
    else: 
        filename = glob.glob(str(filename))[0]

    # read lightning data from file
    #form = 'circle'
    form = 'gauss'
    # data = readLightning (filename, dist=5) ### for plot use this line
    #data = readLightning (filename, IC_CG='CG')
    #data = readLightning (filename, IC_CG='IC')
    dens, densIC, densCG, curr_abs, curr_neg, curr_pos = readLightning (filename, NEAR_REAL_TIME, satscene.time_slot, dt=dt, area=projectionName) # , form='square'

    # get projection 
    projection = pyresample.load_area(os.path.join(CONFIG_PATH, "areas.def"), projectionName)
    #print projection
    satscene.area = projection
    
    print "... channels to load ", satscene.channels_to_load
    for chn_name in satscene.channels_to_load:
        # Read variable corresponding to channel name
        print "... channel to read ", chn_name
        if chn_name == 'dens':
            #print "lightning.py: ", type(dens)
            #print "lightning.py: ", dens.data.shape
            satscene[chn_name] = dens
        elif chn_name == 'densIC':
            satscene[chn_name] = densIC
        elif chn_name == 'densCG':
            satscene[chn_name] = densCG
        elif chn_name == 'curr_abs':
            satscene[chn_name] = curr_abs
        elif chn_name == 'curr_neg':
            #curr_neg*=-1
            satscene[chn_name] = curr_neg
        elif chn_name == 'curr_pos':
            satscene[chn_name] = curr_pos
        elif chn_name == 'curr_per_lightning':
            satscene[chn_name] = curr_abs/dens

    satscene.dt = dt

# ---------------------------------------------------------------------------------

def readLightning(file, NEAR_REAL_TIME, time_slot, dt=5, area='ccs4'):
    """ input 
        file           [string]   input file name
        NEAR_REAL_TIME [logical]  True  -> data from near real time folder
                                  False -> data from archive
        time_slot      [datetime] desired time of lightning analysis 
        dt             [float]    period in min a lightning is counted for
        form           [string]   'square' -> lightning is counted in a square
                                  'circle' -> lightning is counted in a circle
    """

    from numpy import genfromtxt, floor, ceil, zeros, nonzero, rot90, flipud
    from ApproxSwissProj import ApproxSwissProj 
    import numpy.ma as ma
    from os import stat

    from mpop.projector import get_area_def
    obj_area = get_area_def(area)

    ni = obj_area.y_size # 640 for ccs4
    nj = obj_area.x_size # 710 for ccs4

    # empty masked array
    dens     = ma.asarray(zeros(shape=(ni,nj)))
    densIC   = ma.asarray(zeros(shape=(ni,nj)))
    densCG   = ma.asarray(zeros(shape=(ni,nj)))
    curr_abs = ma.asarray(zeros(shape=(ni,nj)))
    curr_neg = ma.asarray(zeros(shape=(ni,nj)))
    curr_pos = ma.asarray(zeros(shape=(ni,nj)))

    # read data from file
    if file != "" and stat(file).st_size != 0:
        
        if NEAR_REAL_TIME:
            # lat        lon    time           intra  current
            # 43.2540    9.9290 20150201033521 IC     35.2
            #                   12345678901234
            data = genfromtxt(file, dtype=None, names=('lat', 'lon', 'time', 'intra', 'current'))  # automatic specification of the format
            if data['time'].size == 1:   # here I get an error for the implecit loop: ??? TypeError: iteration over a 0-d array ???
                years  = [str(data['time'])[ 0: 4]]
                months = [str(data['time'])[ 4: 6]]
                days   = [str(data['time'])[ 6: 8]]
                hours  = [str(data['time'])[ 8:10]]
                mins   = [str(data['time'])[10:12]]
                secs   = [str(data['time'])[12:14]]
                intra   = [data['intra']]   # just a string, instead of an one element string array
                current = [data['current']] # just a string, instead of an one element string array

            else:
                years  = [str(date)[ 0: 4] for date in data['time']]
                months = [str(date)[ 4: 6] for date in data['time']]
                days   = [str(date)[ 6: 8] for date in data['time']]
                hours  = [str(date)[ 8:10] for date in data['time']]
                mins   = [str(date)[10:12] for date in data['time']]
                secs   = [str(date)[12:14] for date in data['time']]
                intra   = data['intra']   
                current = data['current']
            #intra_cloud = zeros(len(data['type']), dtype=bool) # first initialize all with False == cloud to ground lightning
            #intra_cloud[nonzero(data['type'])] = True  
        else: 
            # date                     |  lon   |  lat   |  curr|nS|mode|intra|    Ax |   Ki2|   Exc|      Incl|    Arc|  empty|  empty|  empty|  empty|
            # 01.09.2013 12:19:39.0 UTC|  12.824|  46.506| -16.0| 1|   6|    0|    0.1|   0.4|   1.0|       145|    1.0|    0.0|    0.0|    0.0|    0.0|
            data = genfromtxt(file, dtype=None, names=('date', 'lon', 'lat','current','nS','mode','intra','Ax','Ki2','Exc','Incl','Arc','d1','d2','d3','d4','d5'), delimiter="|") # automatic specification of the format
            days   = [str(date)[ 0: 2] for date in data['date']]
            months = [str(date)[ 3: 5] for date in data['date']]
            years  = [str(date)[ 6:10] for date in data['date']]
            hours  = [str(date)[11:13] for date in data['date']]
            mins   = [str(date)[14:16] for date in data['date']]
            secs   = [str(date)[17:21] for date in data['date']]
            intra   = data['intra']   
            current = data['current']

        #if area=='ccs4':
        if False:
            # define class for reprojection to ccs4 grid 
            ccs4 = ApproxSwissProj()

            # for Bern 
            # data['lat'] = 46.9479
            # data['lon'] = 7.4446
 
            # position in km (x->[-160:480], y->[255,965]
            xCH = ccs4.WGStoCHx(data['lat'],data['lon']) # and x in the vertical         # for Bern 199646.19
            yCH = ccs4.WGStoCHy(data['lat'],data['lon']) # y is in horizontal direction  # for Bern 600454.29

            ## convert to index (i->[0,640], j->[0,710]
            #iCH = floor(yCH/1000.)-255  # y is in horizontal direction
            #jCH = 710-(480-floor(xCH/1000.))  # and x in the vertical ## 
            # convert to index (i->[0,640], j->[0,710]
            iCH = (480-ceil(xCH/1000.))  ## 710- ## x/i is in vertical direction         # for Bern 280.0
            jCH = floor(yCH/1000.)-255    # y/j is in horizontal dir.                    # for Bern 345.0
        else:
            print "... new experimental reprojection"

            if True:
                (jCH, iCH) = obj_area.get_xy_from_lonlat(data['lon'], data['lat'], outside_error=False)       # for Bern (345, 280)
                # returns an int, if only one element lon and lat data is passed
                if isinstance(jCH, int):
                    jCH = [jCH]
                if isinstance(iCH, int):
                    iCH = [iCH]
            else:
                from pyproj import Proj
                p = Proj(obj_area.proj4_string)
                (yCH, xCH) = p(data['lon'], data['lat'])                                 # for Bern (600381.87, 199499.18)
                y_offset = obj_area.area_extent[3]/obj_area.pixel_size_y                 #                 for ccs4 480.0
                iCH = y_offset - ceil(xCH/1000.)                                         # for Bern 280.0
                x_offset = obj_area.area_extent[0]/obj_area.pixel_size_x                 #                 for ccs4 255.0 
                jCH = floor(yCH/1000.) - x_offset                                        # for Bern 345.0

        # considered time period 
        dtime = datetime.timedelta(minutes=dt)

        print "--- lightning time        desired time slot     dtime    intra cloud     current[kA]"
        for i, j, YYYY, MM, DD, hh, mm, ss, ltype, curr in zip(iCH, jCH, years, months, days, hours, mins, secs, intra, current):
            t_light = datetime.datetime(int(YYYY), int(MM), int(DD), int(hh), int(mm), int(float(ss)) ) 
            if ( t_light < time_slot and time_slot-t_light < dtime ):
                #print "   ", str(t_light), " ", str(time_slot), " ", str(time_slot-t_light), "   ", ltype, "          ", curr
                if 0<i and i<ni and 0<j and j<nj:
                    dens[i,j]+=1
                    curr_abs[i,j]+=abs(curr)
                    if curr < 0:
                        curr_neg[i,j]+=curr
                    else:
                        curr_pos[i,j]+=curr
                    if ltype == 'IC' or ltype == 1:
                        #print "... only intra clouds lightnings", YYYY, MM, DD, hh, mm, ltype
                        densIC[i,j]+=1  
                    if ltype=='CG' or ltype==0:
                        #print "... only clouds2ground lightnings", YYYY, MM, DD, hh, mm, ltype
                        densCG[i,j]+=1
            if time_slot + dtime < t_light :
                break
    else:
        print '*** Warning, empty lightning input file '

    print '... dens ndim (lightning.py): ', dens.ndim
    print '... shape (lightning.py): ', dens.shape
    print '... min/max (lightning.py): ', dens.min(), dens.max()

    return dens, densIC, densCG, curr_abs, curr_neg, curr_pos

# ---------------------------------------------------------------------------------

def add_lightning(prop, i, j, dx, form):
    """ add one lightning as data field to the ccs4 array
        add_lightning(dens, i, j, dx, form)
        form           [string]   'square' -> lightning is counted in a square
                                  'circle' -> lightning is counted in a circle
    """

    # print "add lightning at", i, j
    if form == 'square':
        # mark a square 
        prop[i-dx:i+dx,j-dx:j+dx]+=1
    elif form == 'circle':
        from numpy import arange, meshgrid
        # mark a circle 
        rr=dx  # radius
        x = arange(0, 710)  # 709-
        y = arange(0, 640)
        X, Y = meshgrid(x, y)
        #print X
        #print Y        
        #print X.shape
        #print Y.shape
        #print lightnings.shape
        interior = ((X-j)**2 + (Y-i)**2) < rr**2
        prop+=1*interior
    else:
        print '*** ERROR in add_lightning (lightning.py)'
        print '    unknown form ', form

# ---------------------------------------------------------------------------------

def unfold_lightning(prop, dx, form):
    """ unfold one lightning in such a way that it effects dx km
        form           [string]   'square' -> lightning is counted in a square
                                  'circle' -> lightning is counted in a circle
    """

    from scipy.ndimage.filters import convolve
    from numpy import zeros, ones
    import numpy.ma as ma


    # print "unfold_lightning at", i, j
    if form == 'square':
        if dx != 1:
            print "... counting lightning per square with edge lengths of ", dx, " km"
            # mark a square 
            k = ones([2*dx-1,2*dx-1])
            return ma.asarray( convolve(prop, k, mode="constant", cval=0) ) 

    elif form == 'circle':
        if dx != 1:
            print "... counting lightning inside a circle with a radius of ", dx, " km"
            from numpy import arange, meshgrid
            # mark a circle 
            x = arange(0, 2*dx-1) 
            X, Y = meshgrid(x, x)
            k = 1 * ((X-(dx-1))**2 + (Y-(dx-1))**2) < dx**2
            return ma.asarray( convolve(prop, k, mode="constant", cval=0) ) 

    elif form == 'gauss':
        from scipy import ndimage
        print "unfold_lightning (lighting.py) min/max = ", prop.min(), prop.max()
        print '... apply gauss filter with half width of (0.5*', dx, ") km"
            # tuned that it looks similar to the circle with the same dx
        data = ma.asarray( ndimage.gaussian_filter(prop, 0.5*dx, output=np.float32) )
        #print "unfold_lightning (lighting.py) min/max = ", data.min(), data.max()
        d_min=0.001
        print "... neglect small values below ", d_min
        data=ma.masked_less(data, d_min)
        #print "unfold_lightning (lighting.py) min/max = ", data.min(), data.max()
        return data

    elif form == 'gaussgauss':
        from scipy import ndimage
        #print "unfold_lightning (lighting.py) min/max = ", prop.min(), prop.max()
        print '... apply DOUBLE gauss filter with half width of (0.5*', dx, ") km"
            # tuned that it looks similar to the circle with the same dx
        prop = ndimage.gaussian_filter(prop, 0.5*dx)
        data = ma.asarray( ndimage.gaussian_filter(prop, 0.5*dx, output=np.float32) )
        #print "unfold_lightning (lighting.py) min/max = ", data.min(), data.max()
        d_min=0.001
        print "... neglect values below ", d_min
        data=ma.masked_less(data, d_min)
        #print "unfold_lightning (lighting.py) min/max = ", data.min(), data.max()
        return data


    else:
        print '*** ERROR in unfold_lightning (lightning.py)'
        print '    unknown form ', form

def convert_THXprod_nc_daily(input_file_path, output_file_path, date_obj, dt=5, area='ccs4'):
    """Read THX lightning data file and save as daily nc-file.

    Parameters
    ----------
    
    input_file_path : str
        File path defining location of .prod file to be read.
        
    output_file_path : str
        File path defining location of .nc file to be written.
    
    date_obj : list of datetime object
        Datetime object defining date for which to write daily nc-file.
        
    dt : int
        Temporal length in minutes over which lightnings should be added.
    """
    
    if area!='ccs4':
        print("*** Function only implemented for ccs4 projection - no return")
    elif len(date_obj)<1:
        print("*** Function needs at least one datetime object - no return")
        return
    
    while 60%dt!=0:
        dt = int(input("*** 60min modulo dt must be 0, enter new dt: ***"))
    n_temporal_steps = 24*60/dt
        
    ## Loop over all days provided
    for dat in date_obj:
        dat_str = dat.strftime("THX%y%j0000")
        input_file_path_str  = "%s/%s.prd" % (input_file_path,dat_str)
        output_file_path_str = "%s/%s.nc" % (input_file_path,dat_str)
        print("Convert THX ascii file\n   %s\nto\n   %s" %
              (input_file_path_str,output_file_path_str))
        
        ## Predefine numpy array for results:
        THX_array = np.zeros((n_temporal_steps,6,640,710))*np.nan
        
        ## Set hours, minutes etc to zero:
        t0 = dat.replace(hour=0, minute=0)
        t0_list = []
        
        ## Read lightning THX text file:
        for i in range(n_temporal_steps):
            t0 = t0 + datetime.timedelta(minutes=dt)
            t0_list.append(t0)
            if t0.minute==0: print("   ... working on %s" % datetime.datetime.strftime(t0, "%d.%m.%Y %H:%M"))
            THX_tuple = readLightning(file=input_file_path_str, NEAR_REAL_TIME=False,
                                      time_slot=t0, dt=dt, area='ccs4')
            THX_array[i,:,:,:] = np.array(THX_tuple)
        
        ## Save as nc-file:
        THX_nc(output_file_path_str,THX_array,dat,dat_str,n_temporal_steps)
        


def convert_THXprod_nc_daily_direct(input_file_path, output_file_path, date_obj, dt=5, area='ccs4'):
    """Read THX lightning data file (directly) and save as daily nc-file
    This routine does not use the function "readLightning" for every new line, but has
    the essential code included, thus is much more efficient (no opening/closing every
    line).

    Parameters
    ----------
    
    input_file_path : str
        File path defining location of .prod file to be read.
        
    output_file_path : str
        File path defining location of .nc file to be written.
    
    date_obj : list of datetime object
        Datetime object defining date for which to write daily nc-file.
        
    dt : int
        Temporal length in minutes over which lightnings should be added.
    """
    from numpy import genfromtxt, floor, ceil, zeros, nonzero, rot90, flipud
    from ApproxSwissProj import ApproxSwissProj 
    import numpy.ma as ma
    from os import stat
    import pdb
    
    from mpop.projector import get_area_def
    obj_area = get_area_def(area)
    ni = obj_area.y_size # 640 for ccs4
    nj = obj_area.x_size # 710 for ccs4


    if area!='ccs4':
        print("*** Function only implemented for ccs4 projection - no return")
    elif len(date_obj)<1:
        print("*** Function needs at least on datetime object - no return")
        return
    
    while 60%dt!=0:
        dt = int(input("*** 60min modulo dt must be 0, enter new dt: "))
    n_temporal_steps = 24*60/dt
    dtime = datetime.timedelta(minutes=dt)
    
        
    ## Loop over all days provided
    for dat in date_obj:
        t1 = datetime.datetime.now()
        dat_str = dat.strftime("THX%y%j0000")
        input_file_path_str  = "%s/%s.prd" % (input_file_path,dat_str)
        output_file_path_str = "%s/%s.nc" % (output_file_path,dat_str)
        print("Convert THX ascii file\n   %s\nto\n   %s" %
              (input_file_path_str,output_file_path_str))
        
        ## Set hours, minutes etc to zero:
        t0 = dat.replace(hour=0, minute=0)
        temporal_steps = t0 + datetime.timedelta(minutes=dt)*np.arange(n_temporal_steps)
        
        ## Predefine numpy array for results:
        THX_array_dens = np.zeros((n_temporal_steps,3,640,710),dtype=np.uint16)
        THX_array_curr = np.zeros((n_temporal_steps,3,640,710),dtype=np.float32)
        #THX_array = np.zeros((n_temporal_steps,640,710),dtype=np.uint16)
        
        ## Check that ascii file is not empty:
        if os.stat(input_file_path_str).st_size==0:
            print("   THX .prd file is empty, returning zero-array.")
            THX_nc(output_file_path_str,THX_array_curr,THX_array_dens,
                   dat,dat_str,n_temporal_steps,dt,temporal_steps)
            t2 = datetime.datetime.now()
            print("  Elapsed time for NetCDF conversion: "+str(t2-t1)+"\n")
            continue
        
        ## If file is not empty, read it:
        data = genfromtxt(input_file_path_str, dtype=None,
                          names=('date', 'lon', 'lat','current','nS','mode',
                                 'intra','Ax','Ki2','Exc','Incl','Arc','d1',
                                 'd2','d3','d4','d5'), delimiter="|")
        
        ## NOT SO NICE: If ascii file has only one line (one lightning detected),
        ## still return zero array.
        if data['date'].ndim==0:
            print("   THX .prd file has only one line, returning zero-array.")
            THX_nc(output_file_path_str,THX_array_curr,THX_array_dens,
                   dat,dat_str,n_temporal_steps,dt,temporal_steps)
            t2 = datetime.datetime.now()
            print("  Elapsed time for NetCDF conversion: "+str(t2-t1)+"\n")
            continue
            #days    = str(data['date'])[ 0: 2]
            #months  = str(data['date'])[ 3: 5]
            #years   = str(data['date'])[ 6:10] 
            #hours   = str(data['date'])[11:13]
            #mins    = str(data['date'])[14:16]
            #secs    = str(data['date'])[17:21]
            #intra   = data['intra']   
            #current = data['current']
        else:
            days    = [str(date)[ 0: 2] for date in data['date']]
            months  = [str(date)[ 3: 5] for date in data['date']]
            years   = [str(date)[ 6:10] for date in data['date']]
            hours   = [str(date)[11:13] for date in data['date']]
            mins    = [str(date)[14:16] for date in data['date']]
            secs    = [str(date)[17:21] for date in data['date']]
            intra   = data['intra']   
            current = data['current']
        
        ## Reproject lat/lon- to ccsi4 x/y-coordinates
        (jCH, iCH) = obj_area.get_xy_from_lonlat(data['lon'], data['lat'], outside_error=False) #, outside_error=True) # EDIT 3       # for Bern (345, 280)
        
        # returns an int, if only one element lon and lat data is passed
        if isinstance(jCH, int):
            jCH = [jCH]
        if isinstance(iCH, int):
            iCH = [iCH]
            
        ## Read lightning THX text file:
        for i, j, YYYY, MM, DD, hh, mm, ss, ltype, curr in zip(iCH, jCH, years, months, days, hours, mins, secs, intra, current):
            t_light = datetime.datetime(int(YYYY), int(MM), int(DD), int(hh), int(mm)) #, int(float(ss)) ) 
            #if ( t_light < t0 and t0-t_light < dtime ):
            t_light_round = t_light - datetime.timedelta(minutes=t_light.minute % 5)
            if 0<i and i<ni and 0<j and j<nj:
                ti = np.where(temporal_steps==t_light_round)
                #THX_array[ti,i,j]+=1
                THX_array_dens[ti,0,i,j]+=1
                THX_array_curr[ti,0,i,j]+=abs(curr)
                if curr < 0:
                    THX_array_curr[ti,1,i,j]+=curr
                else:
                    THX_array_curr[ti,2,i,j]+=curr
                if ltype == 'IC' or ltype == 1:
                    #print "... only intra clouds lightnings", YYYY, MM, DD, hh, mm, ltype
                    THX_array_dens[ti,1,i,j]+=1  
                if ltype=='CG' or ltype==0:
                    #print "... only clouds2ground lightnings", YYYY, MM, DD, hh, mm, ltype
                    THX_array_dens[ti,2,i,j]+=1
            #if t0 + dtime < t_light :
            #    break
        
        
        #import pysteps as stTHX_array_dens
        #st.plt.animate(THX_array_dens[200:1000,0,:,:],1)
        #pdb.set_trace()
        
        ## Save as nc-file:
        THX_nc(output_file_path_str,THX_array_curr,THX_array_dens,dat,dat_str,n_temporal_steps,dt,temporal_steps)
        t2 = datetime.datetime.now()
        print("  Elapsed time for NetCDF conversion: "+str(t2-t1)+"\n")

        
def THX_nc(output_file_path_str,THX_array_curr,THX_array_dens,dat,dat_str,
           n_temporal_steps,dt,temporal_steps):
    """NetCDF file writer for convert_THXprod_nc_daily_direct function.

    Parameters
    ----------
    
    THX_array : numpy array
        Numpy array with THX values.
    
    output_file_path_str : str
        File path defining location of .nc file to be written.
    
    dat : list of datetime object
        Datetime object defining date for which to write daily nc-file.
        
    dat_str : str
        Date string of dat.
        
    n_temporal_steps : [int]
        Number of time steps (=24h*60min/dt).
    """
    
    from netCDF4 import Dataset, num2date, date2num
    dataset = Dataset(output_file_path_str,
                      'w', format='NETCDF4_CLASSIC')
                      
    ## Dimension creation  
    x = dataset.createDimension('x', THX_array_dens.shape[3])
    y = dataset.createDimension('y', THX_array_dens.shape[2])
    time = dataset.createDimension('time', None) #THX_array_dens.shape[0])
    
    ## Variable creation
    times = dataset.createVariable('time', np.int16, ('time',))
    x_axis = dataset.createVariable('x', np.float32,('x',))
    y_axis = dataset.createVariable('y', np.float32,('y',))
            
    dens     = dataset.createVariable('THX_dens', np.int16,('time','y','x'),zlib=True)
    densCG   = dataset.createVariable('THX_densCG', np.int16,('time','y','x'),zlib=True)
    densIC   = dataset.createVariable('THX_densIC', np.int16,('time','y','x'),zlib=True)
    curr_abs = dataset.createVariable('THX_curr_abs', np.float32,('time','y','x'),zlib=True)
    curr_pos = dataset.createVariable('THX_curr_pos', np.float32,('time','y','x'),zlib=True)
    curr_neg = dataset.createVariable('THX_curr_neg', np.float32,('time','y','x'),zlib=True)
    
    ## Variable attributes
    times.calendar = 'standard'
    times.units = 'minutes since %s' % dat.strftime("%Y-%m-%d %H:%M:%S")
    #times.calendar = 'gregorian'
    #times.units = 'minutes since %s' % (dat.strftime("%Y:%m:%d %H-%M-%S"))
    datetime_list = dat + np.arange(n_temporal_steps)*datetime.timedelta(minutes=dt) + datetime.timedelta(minutes=5)    
    times[:] = date2num(datetime_list,units=times.units) #np.arange(n_temporal_steps)*dt+5
    #times.units = str('%sminutes since %s' % (str(dt), dat.strftime("%Y:%m:%d %H-%M-%S")))
    #times[:] = date2num(temporal_steps, units=times.units, calendar=times.calendar)

    ## Variable assignment
    y_axis.units = 'Swiss northing CH1903 [km]'
    x_axis.units = 'Swiss easting CH1903 [km]'
    #y_axis.units   = 'km of ccs4 projection'
    #x_axis.units   = 'km of ccs4 projection'
    x_axis[:]    = np.arange(255,965)+0.5
    y_axis[::-1] = np.arange(-160,480)+0.5

    dens.setncatts({'long_name': u"Lightning density",'units': u"km-2"})
    densCG.setncatts({'long_name': u"Lightning density (cloud-to-ground)",'units': u"km-2"})
    densIC.setncatts({'long_name': u"Lightning density (Inter/Intra-cloud)",'units': u"km-2"})
    curr_abs.setncatts({'long_name': u"Absolute current",'units': u"kA"})
    curr_pos.setncatts({'long_name': u"Negative current",'units': u"kA"})
    curr_neg.setncatts({'long_name': u"Positive current",'units': u"kA"})
    
    #dens.units = densCG.units = densIC.units = 'km-2' # = 'count'#
    #curr_abs.units = curr_pos.units = curr_neg.units = 'kA'
    dens[:,:,:]     = THX_array_dens[:,0,:,:]
    densIC[:,:,:]   = THX_array_dens[:,1,:,:]
    densCG[:,:,:]   = THX_array_dens[:,2,:,:]
    curr_abs[:,:,:] = THX_array_curr[:,0,:,:]
    curr_neg[:,:,:] = THX_array_curr[:,1,:,:]
    curr_pos[:,:,:] = THX_array_curr[:,2,:,:]
    print("   Max density value = %d" % np.max(dens))
        
    dataset.description = 'THX Lightning data of day '+dat.strftime("%d.%m.%Y")
    dataset.history = 'Created ' + datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    
    dataset.close()
    print("   Written NetCDF file for: %s" % dat.strftime("%d.%m.%y"))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
