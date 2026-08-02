# -*- coding: utf-8 -*-
# Copyright (C) 2022  VT SuperDARN Lab
# Full license can be found in LICENSE.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""music processing module

A module for running the MUltiple SIgnal Classification (MUSIC) algorithm for the detection of
MSTIDs and wave-like structures in SuperDARN data.

For usage examples, please see the iPython notebooks included in the docs folder of the DaViTPy distribution.

References
----------
See Samson et al. [1990] and Bristow et al. [1994] for details regarding the MUSIC algorithm and SuperDARN-observed MSTIDs.

Bristow, W. A., R. A. Greenwald, and J. C. Samson (1994), Identification of high-latitude acoustic gravity wave sources
    using the Goose Bay HF Radar, J. Geophys. Res., 99(A1), 319-331, doi:10.1029/93JA01470.

Samson, J. C., R. A. Greenwald, J. M. Ruohoniemi, A. Frey, and K. B. Baker (1990), Goose Bay radar observations of Earth-reflected,
    atmospheric gravity waves in the high-latitude ionosphere, J. Geophys. Res., 95(A6), 7693-7709, doi:10.1029/JA095iA06p07693.

Module author:: Nathaniel A. Frissell, Fall 2013

Functions
--------------------------------------------------------------------------------------------------------------------------
getDataSet                  get music data object from music array object
stringify_signal            convert dictionary to a string
stringify_signal_list       convert list of dictionaries into strings
beamInterpolation           interpolate music array object along beams
defineLimits                set limits for chosen data set
checkDataQuality            mark data as bad based on radar operations
applyLimits                 remove data outside of limits
determineRelativePosition   find center of cell in music array object
timeInterpolation           interpolate music array object along time
filterTimes                 calculate time range for data set
detrend                     linear detrend of music array/data object
nan_to_num                  convert undefined numbers to finite numbers
windowData                  apply window to music array object
calculateFFT                calculate spectrum of an object
calculateDlm                calculate the cross-spectral matrix of a musicArray/musicDataObj object.
calculateKarr               calculate the two-dimensional horizontal wavenumber array of a musicArray/musicDataObj object.
simulator                   insert a simulated MSTID into the processing chain.
scale_karr                  scale/normalize kArr for plotting and signal detection.
detectSignals               detect local maxima of signals
add_signal                  add signal to detected signal list
del_signal                  remove signal from detected signal list
--------------------------------------------------------------------------------------------------------------------------

Classes
-----------------------------------------------------------
emptyObj        create an empty object
SigDetect       information about detected signals
musicDataObj    basic container for holding MUSIC data.
musicArray      container object for holding musicDataObj's
filter          a filter object for VT sig/siStruct objects
-----------------------------------------------------------

"""
import numpy as np
import datetime 
import copy
import logging

from pydarn import Re #Earth radius

def getDataSet(dataObj,dataSet='active'):
    """Returns a specified musicDataObj from a musicArray object.  If the musicArray object has the exact attribute
    specified in the dataSet keyword, then that attribute is returned.  If not, all attributes of the musicArray object
    will be searched for attributes which contain the string specified in the dataSet keyword.  If more than one are
    found, the last attribute of a sorted list will be returned.  If no attributes are found which contain the specified
    string, the 'active' dataSet is returned. -

    Parameters
    ----------
    dataObj :  musicArray

    dataSet :  Optional[str]
        which dataSet in the musicArray object to process

    Returns
    -------
    currentData : musicDataObj object

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022

    """
    lst = dir(dataObj)
    if dataSet not in lst:
        tmp = []
        for item in lst:
            if dataSet in item:
                tmp.append(item)
        if len(tmp) == 0:
            dataSet = 'active'
        else:
            tmp.sort()
            dataSet = tmp[-1]

    currentData = getattr(dataObj,dataSet)
    return currentData



def stringify_signal(sig):
    """Method to convert a signal information dictionary into a string.

    Parameters
    ----------
    sig : dict
        Information about a detected signal.

    Returns
    -------
    sigInfo : str
        String representation of the signal information.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    sigInfo = {}
    if 'order' in sig:
        sigInfo['order']    = '%d' % sig['order']                   #Order of signals by strength as detected by image detection algorithm
    if 'kx' in sig:
        sigInfo['kx']       = '%.5f' % sig['kx']
    if 'ky' in sig:
        sigInfo['ky']       = '%.5f' % sig['ky']
    if 'k' in sig:
        sigInfo['k']        = '%.3f' % sig['k']
    if 'lambda' in sig:
        if np.isinf(sig['lambda']):
            sigInfo['lambda'] = 'inf'
        else:
            sigInfo['lambda'] = '%d' % np.round(sig['lambda'])      # km
    if 'lambda_x' in sig:
        if np.isinf(sig['lambda_x']):
            sigInfo['lambda_x'] = 'inf'
        else:
            sigInfo['lambda_x'] = '%d' % np.round(sig['lambda_x'])      # km
    if 'lambda_y' in sig:
        if np.isinf(sig['lambda_y']):
            sigInfo['lambda_y'] = 'inf'
        else:
            sigInfo['lambda_y'] = '%d' % np.round(sig['lambda_y'])      # km
    if 'azm' in sig:
        sigInfo['azm']      = '%d' % np.round(sig['azm'])           # degrees
    if 'freq' in sig:
        sigInfo['freq']     = '%.2f' % (sig['freq']*1000.)          # mHz
    if 'period' in sig:
        sigInfo['period']   = '%d' % np.round(sig['period']/60.)    # minutes
    if 'vel' in sig:
        if np.isinf(np.round(sig['vel'])):
            sigInfo['vel']      = 'Inf'
        else:
            sigInfo['vel']      = '%d' % np.round(sig['vel'])
    if 'area' in sig:
        sigInfo['area']     = '%d' % sig['area']                    # Pixels
    if 'max' in sig:
        sigInfo['max']      = '%.4f' % sig['max']                   # Value from kArr in arbitrary units, probably with some normalization
    if 'maxpos' in sig:
        sigInfo['maxpos']   = str(sig['maxpos'])                    # Index position in kArr of maximum value.
    if 'labelInx' in sig:
        sigInfo['labelInx'] = '%d' % sig['labelInx']                # Label value from image processing
    if 'serialNr' in sig:
        sigInfo['serialNr'] = '%d' % sig['serialNr']                # Label value from image processing

    return sigInfo

def stringify_signal_list(signal_list,sort_key='order'):
    """Method to convert a list of signal dictionaries into strings.

    Parameters
    ----------
    signal_list : list of dict
        Information about a detected signal.
    sort_key : Optional[string]
        Dictionary key to sort on, or None for no sort. 'order' will sort the signal list
        from strongest signal to weakest, as determined by the MUSIC algorithm.

    Returns
    -------
    stringInfo : list of str
        String representation of the signal information.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022

    """
    string_info = []

    if sort_key is not None:
        orders  = [x[sort_key] for x in signal_list]
        orders.sort()

        for order in orders:
            for sig in signal_list:
                if sig[sort_key] == order:
                    string_info.append(stringify_signal(sig))
                    signal_list.remove(sig)
    else:
        for sig in signal_list:
            string_info.append(stringify_signal(sig))

    return string_info

def beamInterpolation(dataObj,dataSet='active',newDataSetName='beamInterpolated',comment='Beam Linear Interpolation'):
    """Interpolates the data in a musicArray object along the beams of the radar.  This method will ensure that no
    rangegates are missing data.  Ranges outside of metadata['gateLimits'] will be set to 0.
    The result is stored as a new musicDataObj in the given musicArray object.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    newDataSetName : Optional[str]
        Name of the new musicDataObj to be created in the current musicArray object as a result of this processing.
    comment : Optional[str]
        String to be appended to the history of this object.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    from scipy.interpolate import interp1d
    currentData = getDataSet(dataObj,dataSet)

    nrTimes = len(currentData.time)
    nrBeams = len(currentData.fov["beams"])
    nrGates = len(currentData.fov["gates"])

    interpArr = np.zeros([nrTimes,nrBeams,nrGates])
    for tt in range(nrTimes):
        for bb in range(nrBeams):
            rangeVec  = currentData.fov["slantRCenter"][bb,:]
            input_x   = copy.copy(rangeVec)
            input_y   = currentData.data[tt,bb,:]

            #If metadata['gateLimits'], select only those measurements...
            if 'gateLimits' in currentData.metadata:
                limits = currentData.metadata['gateLimits']
                gateInx = np.where(np.logical_and(currentData.fov["gates"] >= limits[0],currentData.fov["gates"] <= limits[1]))[0]

                if len(gateInx) < 2: continue
                input_x   = input_x[gateInx]
                input_y   = input_y[gateInx]

            good      = np.where(np.isfinite(input_y))[0]
            if len(good) < 2: continue
            input_x   = input_x[good]
            input_y   = input_y[good]

            intFn     = interp1d(input_x,input_y,bounds_error=False,fill_value=0)
            interpArr[tt,bb,:] = intFn(rangeVec)
    newDataSet = currentData.copy(newDataSetName,comment)
    newDataSet.data = interpArr
    newDataSet.setActive()

def defineLimits(dataObj,dataSet='active',rangeLimits=None,gateLimits=None,beamLimits=None,timeLimits=None):
    """Sets the range, gate, beam, and time limits for the chosen data set. This method only changes metadata;
    it does not create a new data set or alter the data in any way.  If you specify rangeLimits, they will be changed to correspond
    with the center value of the range cell.  Gate limits always override range limits.
    Use the applyLimits() method to remove data outside of the data limits.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    rangeLimits : Optional[iterable]
        Two-element array defining the maximum and minumum slant ranges to use. [km]
    gateLimits : Optional[iterable]
        Two-element array defining the maximum and minumum gates to use.
    beamLimits : Optional[iterable]
        Two-element array defining the maximum and minumum beams to use.
    timeLimits : Optional[iterable]
        Two-element array of datetime.datetime objects defining the maximum and minumum times to use.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    currentData = getDataSet(dataObj,dataSet)
    try:
        if (rangeLimits != None) or (gateLimits != None):
            if (rangeLimits != None) and (gateLimits == None):
                inx = np.where(np.logical_and(currentData.fov["slantRCenter"] >= rangeLimits[0],currentData.fov["slantRCenter"] <= rangeLimits[1]))
                gateLimits = [np.min(inx[1][:]),np.max(inx[1][:])]

            if gateLimits != None:
                rangeMin = int(np.min(currentData.fov["slantRCenter"][:,gateLimits[0]]))
                rangeMax = int(np.max(currentData.fov["slantRCenter"][:,gateLimits[1]]))
                rangeLimits = [rangeMin,rangeMax]

            currentData.metadata['gateLimits']  = gateLimits
            currentData.metadata['rangeLimits'] = rangeLimits

        if beamLimits != None:
            currentData.metadata['beamLimits'] = beamLimits

        if timeLimits != None:
            currentData.metadata['timeLimits'] = timeLimits
    except:
        logging.warning("An error occured while defining limits.  No limits set.  Check your input values.")

def checkDataQuality(dataObj,dataSet='active',max_off_time=10,sTime=None,eTime=None):
    """Mark the data set as bad (metadata['good_period'] = False) if the radar was not operational within the chosen time period
    for a specified length of time.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    max_off_time : Optional[int/float]
        Maximum length in minutes radar may remain off.
    sTime : Optional[datetime.datetime]
        Starting time of checking period.  If None, min(currentData.time) is used.
    eTime : Optional[datetime.datetime]
        End time of checking period.  If None, max(currentData.time) is used.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    currentData = getDataSet(dataObj,dataSet)

    if sTime is None:
        sTime   = np.min(currentData.time)

    if eTime is None:
        eTime   = np.max(currentData.time)

    time_vec    = currentData.time[np.logical_and(currentData.time > sTime, currentData.time < eTime)]
    time_vec    = np.concatenate(([sTime],time_vec,[eTime]))
    max_diff    = np.max(np.diff(time_vec))

    if max_diff > datetime.timedelta(minutes=max_off_time):
        currentData.setMetadata(good_period=False)
    else:
        currentData.setMetadata(good_period=True)

    return dataObj

def applyLimits(dataObj,dataSet='active',rangeLimits=None,gateLimits=None,timeLimits=None,newDataSetName='limitsApplied',comment=None):
    """Removes data outside of the rangeLimits and gateLimits boundaries.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    rangeLimits : Optional[iterable]
        Two-element array defining the maximum and minumum slant ranges to use. [km]
    gateLimits : Optional[iterable]
        Two-element array defining the maximum and minumum gates to use.
    beamLimits : Optional[iterable]
        Two-element array defining the maximum and minumum beams to use.
    timeLimits : Optional[iterable]
        Two-element array of datetime.datetime objects defining the maximum and minumum times to use.
    newDataSetName : Optional[str]
        Name of the new musicDataObj to be created in the current musicArray object as a result of this processing.
    comment : Optional[str]
        String to be appended to the history of this object.  Set to None for the Default comment (recommended).

    Returns
    -------
    newData : musicDataObj
        Processed version of input musicDataObj (if succeeded), or the original musicDataObj (if failed).

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """

    if (rangeLimits != None) or (gateLimits != None) or (timeLimits != None):
        defineLimits(dataObj,dataSet='active',rangeLimits=rangeLimits,gateLimits=gateLimits,timeLimits=timeLimits)

    currentData = getDataSet(dataObj,dataSet)
    try:
        #Make a copy of the current data set.

        commentList = []

        if (('timeLimits' in currentData.metadata) == False and 
            ('beamLimits' in currentData.metadata) == False and 
            ('gateLimits' in currentData.metadata) == False):
            return currentData

        newData     = currentData.copy(newDataSetName,comment)
        #Apply the gateLimits
        if 'gateLimits' in currentData.metadata:
            limits      = currentData.metadata['gateLimits']
            gateInx     = np.where(np.logical_and(currentData.fov["gates"] >= limits[0],currentData.fov["gates"]<= limits[1]))[0]

            newData.data = newData.data[:,:,gateInx]
            newData.fov["gates"] = newData.fov["gates"][gateInx]

            newData.fov["latCenter"]     = newData.fov["latCenter"][:,gateInx] 
            newData.fov["lonCenter"]     = newData.fov["lonCenter"][:,gateInx] 
            newData.fov["slantRCenter"]  = newData.fov["slantRCenter"][:,gateInx] 

            #Update the full FOV.
            #This works as long as we look at only consecutive gates.  If we ever do something where we are not looking at consecutive gates
            #(typically for computational speed reasons), we will have to do something else.
            gateInxFull = np.append(gateInx,gateInx[-1]+1) #We need that extra gate since this is the full FOV.
            newData.fov["latFull"] = newData.fov["latFull"][:,gateInxFull] 
            newData.fov["lonFull"] = newData.fov["lonFull"][:,gateInxFull] 
            newData.fov["slantRFull"] = newData.fov["slantRFull"][:,gateInxFull] 

            commentList.append('gate: %i,%i' % tuple(limits))
            rangeLim = (np.min(newData.fov["slantRCenter"]), np.max(newData.fov["slantRCenter"]))
            commentList.append('range [km]: %i,%i' % rangeLim)

            #Remove limiting item from metadata.
            newData.metadata.pop('gateLimits')
            if 'rangeLimits' in newData.metadata: newData.metadata.pop('rangeLimits')
          
        #Apply the beamLimits.
        if 'beamLimits' in currentData.metadata:
            limits      = currentData.metadata['beamLimits']
            beamInx     = np.where(np.logical_and(currentData.fov["beams"] >= limits[0],currentData.fov["beams"] <= limits[1]))[0]

            newData.data = newData.data[:,beamInx,:]
            newData.fov["beams"] = newData.fov["beams"][beamInx]

            newData.fov["latCenter"]     = newData.fov["latCenter"][beamInx,:] 
            newData.fov["lonCenter"]     = newData.fov["lonCenter"][beamInx,:] 
            newData.fov["slantRCenter"]  = newData.fov["slantRCenter"][beamInx,:] 

            #Update the full FOV.
            #This works as long as we look at only consecutive gates.  If we ever do something where we are not looking at consecutive gates
            #(typically for computational speed reasons), we will have to do something else.
            beamInxFull = np.append(beamInx,beamInx[-1]+1) #We need that extra beam since this is the full FOV.
            newData.fov["latFull"] = newData.fov["latFull"][beamInxFull,:] 
            newData.fov["lonFull"] = newData.fov["lonFull"][beamInxFull,:] 
            newData.fov["slantRFull"] = newData.fov["slantRFull"][beamInxFull,:] 

            commentList.append('beam: %i,%i' % tuple(limits))
            #Remove limiting item from metadata.
            newData.metadata.pop('beamLimits')
        
        #Apply the time limits.
        if 'timeLimits' in currentData.metadata:
            limits      = currentData.metadata['timeLimits']
            timeInx     = np.where(np.logical_and(currentData.time >= limits[0],currentData.time <= limits[1]))[0]

            newData.data = newData.data[timeInx,:,:]
            newData.time = newData.time[timeInx]

            commentList.append('time: '+limits[0].strftime('%Y-%m-%d/%H:%M,')+limits[1].strftime('%Y-%m-%d/%H:%M'))
            #Remove limiting item from metadata.
            newData.metadata.pop('timeLimits')
            
            #Update the history with what limits were applied.
            comment = 'Limits Applied'
            commentStr = '['+newData.metadata['dataSetName']+'] '+comment+': '+'; '.join(commentList)
            key = max(newData.history.keys())
            newData.history[key] = commentStr
            logging.debug(commentStr)

        newData.setActive()
        return newData
    except:
        if hasattr(dataObj,newDataSetName): delattr(dataObj,newDataSetName)
#        print 'Warning! Limits not applied.'
        return currentData

def determineRelativePosition(dataObj,dataSet='active',altitude=250.):
    """Finds the center cell of the field-of-view of a musicArray data object.
    The range, azimuth, x-range, and y-range from the center to each cell in the FOV
    is calculated and saved to the FOV object. The following objects are added to
    dataObj.dataSet:
      fov.relative_centerInx: [beam, gate] index of the center cell
      fov.relative_azm:       Azimuth relative to center cell [deg]
      fov.relative_range:     Range relative to center cell [km]
      fov.relative_x:         X-range relative to center cell [km]
      fov.relative_y:         Y-range relative to center cell [km]

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    altitude : Optional[float]
        altitude added to Re = 6378.1 km [km]

    Returns
    -------
    None

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    from pyDARNmusic.utils import geoPack

    #Get the chosen dataset.
    currentData = getDataSet(dataObj,dataSet)

    #Determine center beam.
    ctrBeamInx  = len(currentData.fov["beams"])/2
    ctrGateInx  = len(currentData.fov["gates"])/2

    currentData.fov["relative_centerInx"] = [ctrBeamInx, ctrGateInx]

    #Set arrays of lat1/lon1 to the center cell value.  Use this to calculate all other positions
    #with numpy array math.

    lat1 = np.zeros_like(currentData.fov["latCenter"])   
    lon1 = np.zeros_like(currentData.fov["latCenter"])   

    lat1[:] = currentData.fov["latCenter"][int(ctrBeamInx),int(ctrGateInx)]
    lon1[:] = currentData.fov["lonCenter"][int(ctrBeamInx),int(ctrGateInx)]

    #Make lat2/lon2 the center position array of the dataset.
    lat2    = currentData.fov["latCenter"]
    lon2    = currentData.fov["lonCenter"]

    #Calculate the azimuth and distance from the centerpoint to the endpoint.
    azm     = geoPack.greatCircleAzm(lat1,lon1,lat2,lon2)
    dist    = (Re + altitude)*geoPack.greatCircleDist(lat1,lon1,lat2,lon2)

    #Save calculated values to the current data object, as well as calculate the
    #X and Y relatvie positions of each cell.
    currentData.fov["relative_azm"]    = azm
    currentData.fov["relative_range"]  = dist
    currentData.fov["relative_x"]      = dist * np.sin(np.radians(azm)) 
    currentData.fov["relative_y"]      = dist * np.cos(np.radians(azm)) 

    return None

def timeInterpolation(dataObj,dataSet='active',newDataSetName='timeInterpolated',comment='Time Linear Interpolation',timeRes=10,newTimeVec=None):
    """Interpolates the data in a musicArray object to a regular time grid.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    newDataSetName : Optional[str]
        Name of the new musicDataObj to be created in the current musicArray object as a result of this processing.
    comment : Optional[str]
        String to be appended to the history of this object.
    timeRes : Optional[float]
        time resolution of new time vector [seconds]
    newTimeVec : Optional[list of datetime.datetime]
        Sequence of datetime.datetime objects that data will be interpolated to.  This overides timeRes.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    
    from scipy.interpolate import interp1d
    from pyDARNmusic.utils import timeUtils 
    currentData = getDataSet(dataObj,dataSet)

    sTime = currentData.time[0]
    sTime = datetime.datetime(sTime.year,sTime.month,sTime.day,sTime.hour,sTime.minute) #Make start time a round time.
    fTime = currentData.time[-1]

    #Create new time vector.
    if newTimeVec == None:
        newTimeVec = [sTime]
        while newTimeVec[-1] < fTime:
            newTimeVec.append(newTimeVec[-1] + datetime.timedelta(seconds=timeRes))

    #Ensure that the new time vector is within the bounds of the actual data set.
    newTimeVec  = np.array(newTimeVec)
    good        = np.where(np.logical_and(newTimeVec > min(currentData.time),newTimeVec < max(currentData.time)))
    newTimeVec  = newTimeVec[good]
    newEpochVec = timeUtils.datetimeToEpoch(newTimeVec)

    #Initialize interpolated data.
    nrTimes = len(newTimeVec)
    nrBeams = len(currentData.fov["beams"])
    nrGates = len(currentData.fov["gates"])

    interpArr = np.zeros([nrTimes,nrBeams,nrGates])

    for rg in range(nrGates):
        for bb in range(nrBeams):
            input_x   = currentData.time[:]
            input_y   = currentData.data[:,bb,rg]

            good      = np.where(np.isfinite(input_y))[0]
            if len(good) < 2: continue
            input_x   = input_x[good]
            input_y   = input_y[good]

            input_x   = timeUtils.datetimeToEpoch(input_x)

            intFn     = interp1d(input_x,input_y,bounds_error=False)#,fill_value=0)
            interpArr[:,bb,rg] = intFn(newEpochVec)
    newDataSet = currentData.copy(newDataSetName,comment)
    newDataSet.time = newTimeVec
    newDataSet.data = interpArr
    newDataSet.setActive()

def filterTimes(sTime,eTime,timeRes,numTaps):
    """The linear filter is going to cause a delay in the signal and also won't get to the end of the signal.
    This function will calcuate the full time period of data that needs to be loaded in order to provide filtered data
    for the event requested.

    Parameters
    ----------
    sTime : datetime.datetime
        Start time of event.
    eTime : datetime.datetime
        End time of event.
    timeRes : float
        Time resolution in seconds of data to be sent to filter.
    numtaps : int
        Length of the filter 

    Returns
    -------
    newSTime, newETime : datetime.datetime, datetime.datetime
        Start and end times of data that needs to be fed into the filter.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    td = datetime.timedelta(seconds=(numTaps*timeRes/2.))
    newSTime = sTime - td
    newETime = eTime + td
    return (newSTime, newETime)


def detrend(dataObj,dataSet='active',newDataSetName='detrended',comment=None,type='linear'):
    """Linearly detrend a data in a musicArray/musicDataObj object.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    newDataSetName : Optional[str]
        Name of the new musicDataObj to be created in the current musicArray object as a result of this processing.
    comment : Optional[str]
        String to be appended to the history of this object.  Set to None for the Default comment (recommended).
    type : Optional[str]
        The type of detrending. If type == 'linear' (default), the result of a linear least-squares fit to data
        is subtracted from data. If type == 'constant', only the mean of data is subtracted.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    from scipy import signal

    currentData = getDataSet(dataObj,dataSet)
    currentData = currentData.applyLimits()

    nrTimes, nrBeams, nrGates = np.shape(currentData.data)

    newDataArr= np.zeros_like(currentData.data)
    for bm in range(nrBeams):
        for rg in range(nrGates):
            try:
                newDataArr[:,bm,rg] = signal.detrend(currentData.data[:,bm,rg],type=type)
            except:
                newDataArr[:,bm,rg] = np.nan
  
    if comment == None:
        comment = type.capitalize() + ' detrend (scipy.signal.detrend)'
      
    newDataSet      = currentData.copy(newDataSetName,comment)
    newDataSet.data = newDataArr
    newDataSet.setActive()

def nan_to_num(dataObj,dataSet='active',newDataSetName='nan_to_num',comment=None):
    """Convert all NANs and INFs to finite numbers using numpy.nan_to_num().

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    newDataSetName : Optional[str]
        Name of the new musicDataObj to be created in the current musicArray object as a result of this processing.
    comment : Optional[str]
        String to be appended to the history of this object.  Set to None for the Default comment (recommended).

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    currentData = getDataSet(dataObj,dataSet)
    currentData = currentData.applyLimits()

    if comment == None:
        comment = 'numpy.nan_to_num'
      
    newDataSet      = currentData.copy(newDataSetName,comment)
    newDataSet.data = np.nan_to_num(currentData.data)
    newDataSet.setActive()

def windowData(dataObj,dataSet='active',newDataSetName='windowed',comment=None,window='hann'):
    """Apply a window to a musicArray object.  The window is calculated using scipy.signal.get_window().

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    newDataSetName : Optional[str]
        Name of the new musicDataObj to be created in the current musicArray object as a result of this processing.
    comment : Optional[str]
        String to be appended to the history of this object.  Set to None for the Default comment (recommended).
    window : Optional[str]
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall,
        barthann, kaiser (needs beta), gaussian (needs std), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation)

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    from scipy import signal

    currentData = getDataSet(dataObj,dataSet)
    currentData = currentData.applyLimits()

    nrTimes, nrBeams, nrGates = np.shape(currentData.data)

    win = signal.get_window(window,nrTimes,fftbins=False)
    newDataArr= np.zeros_like(currentData.data)
    for bm in range(nrBeams):
        for rg in range(nrGates):
            newDataArr[:,bm,rg] = currentData.data[:,bm,rg] * win
  
    if comment == None:
        comment = window.capitalize() + ' window applied (scipy.signal.get_window)'
      
    newDataSet      = currentData.copy(newDataSetName,comment)
    newDataSet.data = newDataArr
    newDataSet.setActive()

def calculateFFT(dataObj,dataSet='active',comment=None):
    """Calculate the spectrum of an object.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    comment : Optional[str]
        String to be appended to the history of this object.  Set to None for the Default comment (recommended).

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    from scipy import fftpack

    currentData = getDataSet(dataObj,dataSet)
    currentData = currentData.applyLimits()

    nrTimes, nrBeams, nrGates = np.shape(currentData.data)

    #Determine frequency axis.
    nyq = currentData.nyquistFrequency()
    freq_ax = np.arange(nrTimes,dtype='f8')
    freq_ax = (freq_ax / max(freq_ax)) - 0.5
    freq_ax = freq_ax * 2. * nyq

    #Use complex64, not complex128!  If you use complex128, too much numerical noise will accumulate and the final plot will be bad!
    newDataArr= np.zeros((nrTimes,nrBeams,nrGates),dtype=np.complex64)
    for bm in range(nrBeams):
        for rg in range(nrGates):
            newDataArr[:,bm,rg] = fftpack.fftshift(fftpack.fft(currentData.data[:,bm,rg])) / np.size(currentData.data[:,bm,rg])
  
    currentData.freqVec   = freq_ax
    currentData.spectrum  = newDataArr

    # Calculate the dominant frequency #############################################
    posFreqInx  = np.where(currentData.freqVec >= 0)[0]
    posFreqVec  = currentData.freqVec[posFreqInx]
    npf         = len(posFreqVec) #Number of positive frequencies

    data        = np.abs(currentData.spectrum[posFreqInx,:,:]) #Use the magnitude of the positive frequency data.

    #Average Power Spectral Density
    avg_psd = np.zeros(npf)
    for x in range(npf): avg_psd[x] = np.mean(data[x,:,:])
    currentData.dominantFreq = posFreqVec[np.argmax(avg_psd)]
    currentData.appendHistory('Calculated FFT')
  
def calculateDlm(dataObj,dataSet='active',comment=None):
    """Calculate the cross-spectral matrix of a musicaArray object. FFT must already have been calculated.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    comment : Optional[str]
        String to be appended to the history of this object.  Set to None for the Default comment (recommended).

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    from ctypes import c_void_p, c_double, c_int, cdll
    from numpy.ctypeslib import ndpointer
    
    currentData = getDataSet(dataObj,dataSet)

    nrTimes, nrBeams, nrGates = np.shape(currentData.data)

    nCells                    = nrBeams * nrGates
    currentData.llLookupTable = np.zeros([5,nCells])
    currentData.Dlm           = np.zeros([nCells,nCells],dtype=np.complex128)

    #Only use positive frequencies...
    posInx = np.where(currentData.freqVec > 0)[0]

    #Explicitly write out gate/range indices...
    
    llList = []
    for gg in range(nrGates):
        for bb in range(nrBeams):
            llList.append((bb,gg))
    

    # import ipdb;ipdb.set_trace()
    for ll in range(nCells):
        llAI  = llList[ll] #access tuple in llList using index ll
        ew_dist           = currentData.fov["relative_x"][llAI] #access field of view value of radar using (beam,gate) tuple
        ns_dist           = currentData.fov["relative_y"][llAI] #access field of view value of radar using (beam,gate) tuple
        currentData.llLookupTable[:,ll]  = [ll, currentData.fov["beams"][llAI[0]], currentData.fov["gates"][llAI[1]],ns_dist,ew_dist] 
        spectL            = currentData.spectrum[posInx,llAI[0],llAI[1]]
        fullSpect         = currentData.spectrum[posInx,:,:].T
        result = (np.sum(np.conj(fullSpect) * spectL, axis=-1))
        currentData.Dlm[ll,0:nCells] = result.reshape((nCells))
        # np.savetxt("MYTRANS.cvs",currentData.Dlm)
        # saf = []
        # for mm in range(nCells):
        #     mmAI  = llList[mm]
        #     spectM          = currentData.spectrum[posInx,mmAI[0],mmAI[1]]
        #     currentData.Dlm[ll,mm] = np.sum(spectL * np.conj(spectM))
        #     # saf.append(np.sum(spectL * np.conj(spectM)))
        
    # np.savetxt("MUS.cvs",currentData.Dlm)

    currentData.appendHistory('Calculated Cross-Spectral Matrix Dlm')

def calculateKarr(dataObj,dataSet='active',kxMax=0.05,kyMax=0.05,dkx=0.001,dky=0.001,threshold=0.15):
    """Calculate the two-dimensional horizontal wavenumber array of a musicArray/musicDataObj object.
    Cross-spectrum array Dlm must already have been calculated.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    kxMax : Optional[float]
        Maximum kx (East-West) wavenumber to calculate [rad/km]
    kyMax : Optional[float]
        Maximum ky (North-South) wavenumber to calculate [rad/km]
    dkx : Optional[float]
        kx resolution [rad/km]
    dky : Optional[float]
        ky resolution [rad/km]
    threshold : Optional[float]
        threshold of signals to detect as a fraction of the maximum eigenvalue

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    currentData = getDataSet(dataObj,dataSet)

    nrTimes, nrBeams, nrGates = np.shape(currentData.data)

    #Calculate eigenvalues, eigenvectors
    eVals,eVecs = np.linalg.eig(np.transpose(dataObj.active.Dlm))

    nkx     = np.ceil(2*kxMax/dkx)
    if (nkx % 2) == 0: nkx = nkx+1
    kxVec  = kxMax * (2*np.arange(nkx)/(nkx-1) - 1)

    nky     = np.ceil(2*kyMax/dky)
    if (nky % 2) == 0: nky = nky+1
    kyVec  = kyMax * (2*np.arange(nky)/(nky-1) - 1)

    nkx = int(nkx)
    nky = int(nky)

    xm      = currentData.llLookupTable[4,:] #x is in the E-W direction.
    ym      = currentData.llLookupTable[3,:] #y is in the N-S direction.

    threshold   = 0.15
    maxEval     = np.max(np.abs(eVals))

    minEvalsInx = np.where(eVals <= threshold*maxEval)[0]
    cnt         = np.size(minEvalsInx)
    maxEvalsInx = np.where(eVals >  threshold*maxEval)[0]
    nSigs       = np.size(maxEvalsInx)

    if cnt < 3:
        logging.warning('Not enough small eigenvalues!')


    logging.info('K-Array: ' + str(nkx) + ' x ' + str(nky))
    logging.info('Kx Max: ' + str(kxMax))
    logging.info('Kx Res: ' + str(dkx))
    logging.info('Ky Max: ' + str(kyMax))
    logging.info('Ky Res: ' + str(dky))
    logging.info('')
    logging.info('Signal Threshold:      ' + str(threshold))
    logging.info('Number of Det Signals: ' + str(nSigs))
    logging.info('Number of Noise Evals: ' + str(cnt))

    logging.info('Starting kArr Calculation...')
    t0 = datetime.datetime.now()
    def vCalc(um,v):
        return np.dot( np.conj(um), v) * np.dot( np.conj(v), um)

    vList = [eVecs[:,minEvalsInx[ee]] for ee in range(cnt)]
    kArr  = np.zeros((nkx,nky),dtype=np.complex64)
    

    import time
    # import ipdb;ipdb.set_trace()
    t0 = time.time()
    fullkxxm = np.multiply.outer(kxVec,xm) #multiples every element in kxVec with every element in xm
    fullkyym = np.multiply.outer(kyVec,ym) #multiples every element in kyVec with every element in ym
    fullumm = np.exp(1j*(fullkxxm+fullkyym[:,None])) #add every element in fullkxxm with every element in fullkyym
    
    reshapefullumm = fullumm.reshape((fullumm.shape[0]*fullumm.shape[1],fullumm.shape[2]),order='F')
    xlength_of_reshapefullumm = reshapefullumm.shape[0]
    conjum = np.zeros((xlength_of_reshapefullumm,len(vList)),dtype=np.complex64)
    conjv = np.zeros((xlength_of_reshapefullumm,len(vList)),dtype=np.complex64)
    
    vList2 = np.array(vList)
    currentIndex = 0
    end = reshapefullumm.shape[0]
    # currentIndex = end - currentIndex
    safeRFullum = reshapefullumm[currentIndex:end,:]
    conjum[currentIndex:end,:] = np.dot(vList2,np.conj(safeRFullum.T)).T
    conjv[currentIndex:end,:] = np.dot(np.conj(vList2), safeRFullum.T).T 
    # import ipdb;ipdb.set_trace()
    kArr = (1. / np.sum((conjum*conjv),axis=1)).reshape(nkx,nky)

    t1 = time.time()
    print(f"PROCESING TIME: {t1-t0}")


    logging.info('Finished kArr Calculation.  Total time: ' + str(t1-t0))

    currentData.karr  = kArr
    currentData.kxVec = kxVec
    currentData.kyVec = kyVec
    currentData.appendHistory('Calculated kArr')


def simulator(dataObj, dataSet='active',newDataSetName='simulated',comment=None,keepLocalRange=True,sigs=None,noiseFactor=0):
    """Replace SuperDARN Data with simulated MSTID(s).  This is useful for understanding how the signal processing
    routines of this module affect ideal data.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    newDataSetName : Optional[str]
        Name of the new musicDataObj to be created in the current musicArray object as a result of this processing.
    comment : Optional[str]
        String to be appended to the history of this object.  Set to None for the Default comment (recommended).
    keepLocalRange : Optional[bool]
        If true, the locations calculated for the actual radar field of view will be used.  If false,
        a linearly-spaced will replace the true grid.
    sigs : Optional[list of tuples]
        A list of tuples defining the characteristics of the simulated signal.  Sample list is as follows.
        If this keyword is None, the values in this sample list are used as the default values.::

        sigs = []
            #           (amp,    kx,      ky,      f, phi, dcOffset)
            sigs.append((  5,  0.01,  -0.010, 0.0004,   0,       5.))
            sigs.append((  5, 0.022,  -0.023, 0.0004,   0,       5.))

        Each signal is evaluated as a cosine and then summed together.  The cosine evaluated is::
        sig  = amp * np.cos(kx*xgrid + ky*ygrid - 2.*np.pi*f*t + phi) + dc
    noiseFactor : Optional[float]
        Add white gaussian noise to the simulated signal.  noiseFactor is a scalar such that:
        noise             = noiseFactor*np.random.standard_normal(nSteps)

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    from pyDARNmusic.utils.timeUtils import datetimeToEpoch
    
    currentData = getDataSet(dataObj,dataSet)

    #Typical TID Parameters:
    #       Frequency:      0.0003 mHz
    #       Period:         55.5 min
    #       H. Wavelength:  314 km
    #       k:              0.02 /km

    if keepLocalRange == True:
        nx, ny  = np.shape(currentData.fov["relative_x"])
        xRange  = np.max(currentData.fov["relative_x"]) - np.min(currentData.fov["relative_x"])
        yRange  = np.max(currentData.fov["relative_y"]) - np.min(currentData.fov["relative_y"])

        xgrid   = currentData.fov["relative_x"]
        ygrid   = currentData.fov["relative_y"]
    else:
        nx      = 16
        xRange  = 800.
        ny      = 25
        yRange  = 600.

        xvec    = np.linspace(-xRange/2.,xRange/2.,nx)
        yvec    = np.linspace(-yRange/2.,yRange/2.,ny)

        dx      = np.diff(xvec)[0]
        dy      = np.diff(yvec)[0]

        xaxis   = np.append(xvec,xvec[-1]+dx)
        yayis   = np.append(yvec,yvec[-1]+dy)

        xgrid   = np.zeros((nx,ny))
        ygrid   = np.zeros((nx,ny))

        for kk in range(nx): ygrid[kk,:] = yvec[:]
        for kk in range(ny): xgrid[kk,:] = yvec[:]

    if sigs == None:
        #Set some default signals.
        sigs = []
        #           (amp,    kx,      ky,      f, phi, dcOffset)
        sigs.append((  5,  0.01,  -0.010, 0.0004,   0,       5.))
        sigs.append((  5, 0.022,  -0.023, 0.0004,   0,       5.))
  
    secVec  = np.array(datetimeToEpoch(currentData.time))
    secVec  = secVec - secVec[0]

    nSteps  = len(secVec)
    dt      = currentData.samplePeriod()

    dataArr = np.zeros((nSteps,nx,ny)) 

    for step in range(nSteps):
        t = secVec[step]
        for kk in range(len(sigs)):
            amp     = sigs[kk][0]
            kx      = sigs[kk][1]
            ky      = sigs[kk][2]
            f       = sigs[kk][3]
            phi     = sigs[kk][4]
            dc      = sigs[kk][5]

            if 1./dt <= 2.*f:
                logging.warning('Nyquist Violation in f.')
                logging.warning('Signal #: %i' % kk)

#            if 1./dx <= 2.*kx/(2.*np.pi):
#                print 'WARNING: Nyquist Violation in kx.'
#                print 'Signal #: %i' % kk
#
#            if 1./dy <= 2.*ky/(2.*np.pi):
#                print 'WARNING: Nyquist Violation in ky.'
#                print 'Signal #: %i' % kk

            temp    = amp * np.cos(kx*xgrid + ky*ygrid - 2.*np.pi*f*t + phi) + dc
            dataArr[step,:,:] = dataArr[step,:,:] + temp

    #Signal RMS
    sig_rms = np.zeros((nx,ny))
    for xx in range(nx):
        for yy in range(ny):
            sig_rms[xx,yy] = np.sqrt(np.mean((dataArr[:,xx,yy])**2.))

    noise_rms = np.zeros((nx,ny))
    if noiseFactor > 0:
        nf = noiseFactor
        #Temporal White Noise
        for xx in range(nx):
            for yy in range(ny):
                noise             = nf*np.random.standard_normal(nSteps)
                noise_rms[xx,yy]  = np.sqrt(np.mean(noise**2))
                dataArr[:,xx,yy]  = dataArr[:,xx,yy] + noise

    xx      = np.arange(ny)
    mu      = (ny-1.)/2.
    sigma2  = 10.0
    sigma   = np.sqrt(sigma2)
    rgDist  = 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((xx-mu)/sigma)**2)
    rgDist  = rgDist / np.max(rgDist)

    mask    = np.zeros((nx,ny))
    for nn in range(nx): mask[nn,:] = rgDist[:]

    mask3d  = np.zeros((nSteps,nx,ny))
    for nn in range(nSteps): mask3d[nn,:,:] = mask[:]

    #Apply Range Gate Dependence
    dataArr = dataArr * mask3d

    snr     = (sig_rms/noise_rms)**2
    snr_db  = 10.*np.log10(snr)

    if comment == None:
        comment = 'Simulated data injected.'
      
    newDataSet      = currentData.copy(newDataSetName,comment)
    newDataSet.data = dataArr
    newDataSet.setActive()

    #OPENW,unit,'simstats.txt',/GET_LUN,WIDTH=300
    #stats$  = ' Mean: '   + NUMSTR(MEAN(sig_rms),3)         $
    #        + ' STDDEV: ' + NUMSTR(STDDEV(sig_rms),3)       $
    #        + ' Var: '    + NUMSTR(STDDEV(sig_rms)^2,3)
    #PRINTF,unit,'SIG_RMS'
    #PRINTF,unit,stats$
    #PRINTF,unit,sig_rms
    #
    #PRINTF,unit,''
    #PRINTF,unit,'NOISE_RMS'
    #stats$  = ' Mean: '   + NUMSTR(MEAN(noise_rms),3)         $
    #        + ' STDDEV: ' + NUMSTR(STDDEV(noise_rms),3)       $
    #        + ' Var: '    + NUMSTR(STDDEV(noise_rms)^2,3)
    #PRINTF,unit,stats$
    #PRINTF,unit,noise_rms
    #
    #PRINTF,unit,''
    #PRINTF,unit,'SNR_DB'
    #stats$  = ' Mean: '   + NUMSTR(MEAN(snr_db),3)         $
    #        + ' STDDEV: ' + NUMSTR(STDDEV(snr_db),3)       $
    #        + ' Var: '    + NUMSTR(STDDEV(snr_db)^2,3)
    #PRINTF,unit,stats$
    #PRINTF,unit,snr_db
    #CLOSE,unit

def scale_karr(kArr):
    from numpy import nanstd, nanmean
    """Scale/normalize kArr for plotting and signal detection.
    
    Parameters
    ----------
    kArr : 2D numpy.array
        Two-dimensional horizontal wavenumber array of a musicArray/musicDataObj object.

    Returns
    -------
    data : 2D numpy.array
        Scaled and normalized version of kArr.

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    data        = np.abs(kArr) - np.min(np.abs(kArr))

    #Determine scale for colorbar.
    scale       = [0.,1.]
    sd          = nanstd(data,axis=None)
    mean        = nanmean(data,axis=None)
    scMax       = mean + 6.5*sd
    data        = data / scMax
    return data

def detectSignals(dataObj,dataSet='active',threshold=0.35,neighborhood=(10,10)):
    """Automatically detects local maxima/signals in a calculated kArr.  This routine uses the watershed
    algorithm from the skimage image processing library.  Results are automatically stored in
    dataObj.dataSet.sigDetect.

    Parameters
    ----------
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    threshold : Optional[float]
        Scaled input data must be above this value to be detected.  A higher number
        will reduce the number of signals detected.
    neighborhood : Optional[tuple]
        Local region in which to search for peaks at every point in the image/array.
        (10,10) will search a 10x10 pixel area.

    Returns
    -------
    currentData : musicDataObj
        object

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    currentData = getDataSet(dataObj,dataSet)
    ################################################################################
    #Feature detection...
    #Now lets do a little image processing...
    from scipy import ndimage
    # from skimage.morphology import watershed
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from pyDARNmusic.music import SigDetect
    #sudo pip install cython
    #sudo pip install scikit-image

    data = scale_karr(currentData.karr)

    mask        = data > threshold
    labels, nb  = ndimage.label(mask)

    distance    = ndimage.distance_transform_edt(mask)

    # indices keyword is deprecated.
#    local_maxi  = peak_local_max(distance,footprint=np.ones(neighborhood),indices=False)

    # Do this instead because indices keyword is deprecated.
    peak_inx    = peak_local_max(distance,footprint=np.ones(neighborhood))
    local_maxi  = np.zeros_like(distance, dtype=bool)
    local_maxi[tuple(peak_inx.T)] = True

    markers,nb  = ndimage.label(local_maxi)
    labels      = watershed(-distance,markers,mask=mask)

    areas         = ndimage.sum(mask,labels,range(1,labels.max()+1))
    maxima        = ndimage.maximum(data,labels,range(1, labels.max()+1))
    order         = np.argsort(maxima)[::-1] + 1
    maxpos        = ndimage.maximum_position(data,labels,range(1, labels.max()+1))

    sigDetect = SigDetect()
    sigDetect.mask    = mask
    sigDetect.labels  = labels
    sigDetect.nrSigs  = nb
    sigDetect.info    = []
    for x in range(labels.max()):
        info = {}
        info['labelInx']    = x+1
        info['order']       = order[x]
        info['area']        = areas[x]
        info['max']         = maxima[x]
        info['maxpos']      = maxpos[x]
        info['kx']          = currentData.kxVec[int(info['maxpos'][0])]
        info['ky']          = currentData.kyVec[int(info['maxpos'][1])]
        info['k']           = np.sqrt( info['kx']**2 + info['ky']**2 )      # Horizontal Wavenumber [1/(2*pi*km)]
        info['lambda_x']    = 2*np.pi / info['kx']                          # North-South Wavelength in km
        info['lambda_y']    = 2*np.pi / info['ky']                          # East-West Wavelenth in km
        info['lambda']      = 2*np.pi / info['k']                           # Horizonal Wavelength in km
        info['azm']         = np.degrees(np.arctan2(info['kx'],info['ky'])) # Propagation azimuth [degrees clockwise from North]
        info['freq']        = currentData.dominantFreq                      # Frequency in Hz
        info['period']      = 1./currentData.dominantFreq                   # Period in seconds
        info['vel']         = (2.*np.pi/info['k']) * info['freq'] * 1000.   # MSTID Velocity in m/s
        sigDetect.info.append(info)

    currentData.appendHistory('Detected KArr Signals')
    currentData.sigDetect = sigDetect
    return currentData

def add_signal(kx,ky,dataObj,dataSet='active',frequency=None):
    """Manually add a signal to the detected signal list.  All signals will be re-ordered according to value in the 
    scaled kArr.  Added signals can be distinguished from autodetected signals because 
    'labelInx' and 'area' will both be set to -1.

    Parameters
    ----------
    kx : float
        Value of kx of new signal.
    ky : float
        Value of ky of new signal.
    dataObj : musicArray
        musicArray object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process
    frequency : Optional[float]
        Frequency to use to calculate period, phase velocity, etc.  If None, 
        the calculated dominant frequency will be used.

    Returns
    -------
    currentData : musicDataObj
        object

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """
    currentData = getDataSet(dataObj,dataSet)
    data = scale_karr(currentData.karr)

    def find_nearest_inx(array,value):
        return (np.abs(array-value)).argmin()

    kx_inx  = find_nearest_inx(currentData.kxVec,kx)
    ky_inx  = find_nearest_inx(currentData.kyVec,ky)

    maxpos      = (kx_inx,ky_inx)
    value       = data[kx_inx,ky_inx]

    true_value  = currentData.karr[kx_inx,ky_inx] #Get the unscaled kArr value.

    if frequency == None:
        freq    = currentData.dominantFreq
    else:
        freq = frequency

    info = {}
    info['labelInx']    = -1
    info['area']        = -1
    info['order']       = -1
    info['max']         = value
    info['true_max']    = true_value    #Unscaled kArr value
    info['maxpos']      = maxpos
    info['kx']          = currentData.kxVec[info['maxpos'][0]]
    info['ky']          = currentData.kyVec[info['maxpos'][1]]
    info['k']           = np.sqrt( info['kx']**2 + info['ky']**2 )
    info['lambda_x']    = 2*np.pi / info['kx']
    info['lambda_y']    = 2*np.pi / info['ky']
    info['lambda']      = 2*np.pi / info['k']
    info['azm']         = np.degrees(np.arctan2(info['kx'],info['ky']))
    info['freq']        = freq
    info['period']      = 1./freq
    info['vel']         = (2.*np.pi/info['k']) * info['freq'] * 1000.

    currentData.sigDetect.info.append(info)
    currentData.sigDetect.reorder()
    currentData.appendHistory('Appended Signal to sigDetect List')

    return currentData

def del_signal(order,dataObj,dataSet='active'):
    """Remove a signal to the detected signal list.

    Parameters
    ----------
    order :
        Single value of list of signal orders (ID's) to be removed from the list.
    dataObj : musicArray
        object
    dataSet : Optional[str]
        which dataSet in the musicArray object to process

    Returns
    -------
    currentData : musicDataObj
        object

    Written by Nathaniel A. Frissell, Fall 2013
    Updated by: Francis Tholley, 2022
    """

    currentData = getDataSet(dataObj,dataSet)
    data = scale_karr(currentData.karr)

    orderArr = np.array(order)

    for item in list(currentData.sigDetect.info):
        if item['order'] in orderArr:
            currentData.sigDetect.info.remove(item)

    currentData.sigDetect.reorder()
    currentData.appendHistory('Deleted Signals from sigDetect List')
    return currentData

