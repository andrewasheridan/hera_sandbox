### HERA LST Aligner
#
# James Kent, University of Cambridge
# jck42@cam.ac.uk
#
# This script takes a directory of closure files outputted from recipe using
# heracasa (Bojan Nikolic, bn204@cam.ac.uk), and from there bins them so that
# closure measurements over successive days are aligned in LST.
#
# VERY MUCH A WORK IN PROGRESS. Plenty to do and tidy up...


from astropy.time import Time
import matplotlib.pyplot as plt
import os
import numpy
import argparse
import textwrap


class LST_Binner(object):

    """
    Class that takes the aligned LSTs and a dictionary of ALL the closures from
    all input days and concatenates them together and outputs them as a
    single numpy array.
    
    Attributes:

    lst_array     [Numpy array] This is an array of all aligned LSTS. We take the columns
                  of this to index {closure_dict} to get the closure phases for the 
                  correct LST.

    closure_dict  [Dictionary] Two-layer dictionary, keyed by date then by LST. 
                  Contains the closure phases as a [Numpy array] of shape 
                  (no_triads,no_channels).

    lst_start     [Integer] Column index for [lst_array] to output binned LST's from.

    lst_range     [Integer] Range of LST's to bin.

    triad_no      [Integer] Number of triads in closure phases in [closure_dict].

    channels      [Integer] Number of channels in closure phases in [closure_dict].

    day_array     [Numpy array] of shape (no_days). Gives UTC Dates.
    
    outlst_array  [Numpy array] of shape (no_lsts,no_days)Exact LST's of closures outputted.

    data_array    [Numpy array] of shape (no_lsts,no_days,no_triads,no_channels). Contains 
                  closure phases.

    flag_array    [Numpy array] of shape (no_lsts,no_days,no_triads,no_channels). Contains
                  individual channel flags from the real time processor(RTP).

    triad_array   [Numpy array] of shape (no_triads,3). Contains antennas which make up the
                  individual triads.

    Member Functions:

    __init__() Initialises an instance of class LST_Binner

    __bin_lsts() Where the magic happens. Takes our aligned LST's and outputs the closures.

    __save_binned_lsts() Saves day_array, outlst_array and data_array to a .npz file.


    TODO: Work out what to do with the bits on the end we don't care about.
    TODO: Averaging    
    """

    def __init__(self,
                 lst_array,
                 closure_dict,
                 lst_start = 0,
                 lst_range = 15,
                 triad_no = 45,
                 channels = 1024):
        """
        Instantiates the LST_Binner class which takes a numpy array of aligned LST's and
        a dictionary describing all of our closure data and outputs the items of interest.

        Inputs:

        lst_array     [Numpy Array] Array of aligned LST's. Individual elements used as 
                      keys for closure_dict.

        closure_dict  [Dictionary] Three-layer dictionary with all the closure information,
                      as well as flags, triads, days etc.

        lst_start     [Integer] Column index for [lst_array] to output binned LST's from.

        lst_range     [Integer] Range of LST's to bin.

        triad_no      [Integer] Number of triads in dataset.

        channels      [Integer] Number of frequency channels.

        """

        #Parameters
        self.lst_array = lst_array
        self.closure_dict = closure_dict
        self.lst_start = lst_start
        self.lst_range = lst_range
        self.triad_no = triad_no
        self.channels = channels
        #Our output arrays for our LST binned data.
        self.day_array = None
        self.outlst_array = None
        self.data_array = None
        self.flag_array = None
        self.triad_array = None
        self.averaged_data_array = None
        self.std_dev_lst = None
        self.std_dev_triad = None

    def __project_unit_circle(self,closures):
        """
        Project angular data onto unit circle / convert to cartesian
        co-ordinates.

        Inputs:

        closures    [Numpy Array] Binned closures of shape [lst, day,
                     triad, channel]

        """

        return numpy.cos(closures),numpy.sin(closures)

    def __stddev_circular(self,closures,axis):
        """
        Finds the standard deviation of the angular data using the defined
        statistics for a von Mises distribution on the unit circle, or a 
        wrapped gaussian.

        Inputs:

        closures    [Numpy Array] Binned closures of shape [lst, day,
                     triad, channel]
        axis        [Integer] Which axis of [closures] to find std_dev over.
        
        """


        closures_x, closures_y = self.__project_unit_circle(closures)
        averaged_x = numpy.mean(closures_x,axis=axis)
        averaged_y = numpy.mean(closures_y,axis=axis)
        r = numpy.sqrt(numpy.square(averaged_x) + numpy.square(averaged_y))
        return numpy.sqrt(-2 * numpy.log(r))

    def __average_circular(self,closures,axis):
        """
        Takes the average angle of angular closure data.

        Inputs:

        closures    [Numpy Array] Binned closures of shape [lst, day,
                     triad, channel]
        axis        [Integer] Which axis of [closures] to average over.
        """

        closures_x, closures_y = self.__project_unit_circle(closures)
        averaged_x = numpy.mean(closures_x,axis=axis)
        averaged_y = numpy.mean(closures_y,axis=axis)
        return numpy.arctan2(averaged_y, averaged_x)
    
    def bin_lsts(self,average=False,std_dev=False):
        
        """
        Takes our binned LST_Array and uses it to index our dictionary of all
        closures. Then extracts the LST's of interest and saves them to 
        day_array, outlst_array, data_array. See class description.

        Inputs:

        average   [Bool] Wether to average across LST's or not.
        std_dev   [Bool] Wether to calculate std_deviations across both
                  LST's and Triads(to check redundancy) or not.
        """

        # Describes the days in our LST bins
        self.day_array = numpy.zeros(shape=len(self.closure_dict.keys()))
        # Gives exact LST's for our aligned bins (for reference)
        self.outlst_array = numpy.zeros(shape=(self.lst_range, len(self.closure_dict.keys())))
        # Final concatenated closure phases, aligned by LST
        self.data_array = numpy.zeros(shape=(self.lst_range,
                                        len(self.closure_dict.keys()),
                                        self.triad_no,
                                        self.channels))
        # All of the RTP flags, aligned with closure phases.
        self.flag_array = numpy.zeros(shape=(self.lst_range,
                                        len(self.closure_dict.keys()),
                                        self.triad_no,
                                        self.channels),
                                      dtype=numpy.int8)
        # Our triads.
        self.triad_array = numpy.zeros(shape=(self.triad_no, 3))
        
                                        
        xvar = numpy.arange(0,1024)
        for i, date in enumerate(sorted(self.closure_dict.keys())):
            self.day_array[i] = int(date)

            for lst in range(self.lst_range):
                
                
                lst_index = self.lst_array[i,self.lst_start+lst]
                self.outlst_array[lst,i] = lst_index 
                closures_at_lst = self.closure_dict[date][lst_index]['phase']
                flags_at_lst = self.closure_dict[date][lst_index]['flags']
                
                #print(numpy.shape(closures_at_lst))
                #print(lst_index)

                self.data_array[lst,i,:,:] = closures_at_lst
                self.flag_array[lst,i,:,:] = flags_at_lst
                self.triad_array = self.closure_dict[date][lst_index]['triads']

        print(self.outlst_array)
        if average == True:
            self.averaged_data_array = self.__average_circular(self.data_array,1)

        if std_dev == True:
            self.std_dev_lst = self.__stddev_circular(self.data_array,1)
            self.std_dev_triad = self.__stddev_circular(self.data_array,2)

        

    def save_binned_lsts(self,filename):

        """
        Takes our binned closures and outputs them to a .npz file.

        Inputs:

        filename    [String] Filename to save .npz file.
        """

        if self.day_array is not None:
            if (self.averaged_data_array is None) and (self.std_dev_lst is None):
                numpy.savez(filename,
                            days=self.day_array,
                            last=self.outlst_array,
                            closures=self.data_array,
                            flags=self.flag_array,
                            triads=self.triad_array)
            elif (self.averaged_data_array is not None) and (self.std_dev_lst is None):
                numpy.savez(filename,
                            days=self.day_array,
                            last=self.outlst_array,
                            closures=self.data_array,
                            flags=self.flag_array,
                            triads=self.triad_array,
                            averaged_closures=self.averaged_data_array)
            elif (self.averaged_data_array is None) and (self.std_dev_lst is not None):
                numpy.savez(filename,
                            days=self.day_array,
                            last=self.outlst_array,
                            closures=self.data_array,
                            flags=self.flag_array,
                            triads=self.triad_array,
                            std_dev_lst=self.std_dev_lst,
                            std_dev_triad=self.std_dev_triad)
            else:
                numpy.savez(filename,
                            days=self.day_array,
                            last=self.outlst_array,
                            closures=self.data_array,
                            flags=self.flag_array,
                            triads=self.triad_array,
                            averaged_closures=self.averaged_data_array,
                            std_dev_lst=self.std_dev_lst,
                            std_dev_triad=self.std_dev_triad)
                            
            
        
        else:
            raise ValueError("LST's not binned")




class LST_Alignment(object):
    """
    Class takes a dictionary of datestamps and aligns them to each other.
    We take advantage of sidereal time moving 235.9 seconds per day with respect
    to the solar day.

    Attributes:

    closure_directory      [String] Base directory where heracasa .npz files are located.
    
    date_set               [List] Ordered dates , earliest -> latest
    
    date_dict              [Dictionary] Dictionary of all dates with filenames.

    timestamp_delta        [Float] Delta between sidereal day and solar day.

    delta_ind              [Int] How many timestamp indices drift per sidereal day.

    delta_rem              [Float] Remainder from delta_ind calculation.

    Member Functions:

    __init__()             Initialises an instance of class LST_Alignment.

    __extract_closures()   Opens all the .npz files and concatenates them into a single dictionary 
                           keyed by date/lst.

    __align_closures()     Aligns the closures by LST.

    align_timestamps()     Public function which builds the date_dict and aligns the LST's, and 
                           returns them.

    TODO: Implement destructive alignment.
    TODO: Implement the flags?
    TODO: Package triad information through as well?
    TODO: ^^^^^ Fix Alignment before these ^^^
    """

    def __init__(self,
                 closure_directory, 
                 ordered_dates, 
                 date_dict, 
                 integration_time=10.7, 
                 sidereal_delta=235.909, 
                 destructive = True): 
        """
        Initialises the LST_Alignment Class which aligns the LST's over successive
        Julian Dates"

        Inputs:

        closure_directory   [String] Root directory of heracasa .npz files.
        
        ordered_dates       [List] All datestamps in order. Earliest -> Latest

        date_dict           [Dictionary] of all dates with filenames.

        integration_time    [Float] HERA Integration Time. Default = 10.7s.

        sidereal_delta      [Float] Delta between sidereal day and solar day.
        
        destructive         [Bool] Destructive alignment or not? [NOT IMPLEMENTED (YET)]
        """
        self.closure_directory = closure_directory
        self.date_set = ordered_dates
        self.date_dict = date_dict
        self.timestamp_delta = sidereal_delta / integration_time
        self.delta_ind = int(round(self.timestamp_delta)) #Get closest indice
        self.delta_rem = self.timestamp_delta % 1


    def __extract_closures(self):

        """
        Opens all of the .npz files in date_dict, and sorts them into a new dictionary
        which is keyed by both date and LST. Thus the key mechanism is as so:

        |
        | - Date_1 - LST_1 - 'phase' - [Numpy Array]
        |                  - 'flags' - [Numpy Array]
        |                  - 'triads' - [Numpy Array]
        |	   - LST_2	
        |
        | - Date_2 - LST_1
        |
        | - etc

        """

        closure_dict = {}
     #Only works in Python 2.x
        for date, npz_files in sorted(self.date_dict.iteritems()):
            closure_dict[date] = {}
            for npz_file in npz_files:
                with numpy.load(self.closure_directory + npz_file) as data:

                    # LST's is used to build the keys of the second tree layer,
                    # as we want to organise by LST for our alignment.
                    lsts = data['LAST']
                    
                    phases = data['phase']
                    flags = data['flags']
                    triads = data['tr']
                    for i,lst in enumerate(lsts):
                        closure_dict[date][lst] = {} 
                        closure_dict[date][lst]['phase'] = phases[:,0,:,i]
                        closure_dict[date][lst]['flags'] = flags[:,0,:,i]
                        closure_dict[date][lst]['triads'] = triads
                        
        return(closure_dict)

    def __align_closures(self, reference_lst, closures):

        """
        Does the alignment of the LST's over successive Julian Days. 
        Little bit shakey and basic but works okay for now.

        Inputs:

        reference_lst  [Dictionary] First LST in dataset for reference. 
                       Can get rid I think.

        closures       [Dictionary] Dictionary of all closures.


        TODO: Tidy this up, reference_lst not needed?
        """

        #Generate Numpy array from closure_dictionary. Makes life significantly easier.
        lst_array = numpy.zeros(shape=(len(closures.keys()),len(reference_lst)))
        i = 0
        initial_date, initial_lsts = sorted(closures.iteritems())[0]
        for date, lst_s in sorted(closures.iteritems()):
            
            for j,lst in enumerate(sorted(lst_s)):
                lst_array[i,j] = lst
            i = i + 1
        
        print(lst_array)

        #Align LST's.

        offset_array = numpy.zeros(shape=(len(closures.keys())))
        datelist = reversed(self.date_set)
        print(datelist)
        prev_date = None
        for i, date in enumerate(reversed(self.date_set)):

            if i == 0:
                offset_array[i] = 0
            else:
                date_delta = int(prev_date) - int(date)
                offset_array[i] = offset_array[i-1] - (self.delta_ind - self.delta_rem)  * date_delta
                
            prev_date = date
        offset_array = numpy.rint(offset_array)
        offset_array = numpy.flipud(offset_array)
        offset_array = offset_array.astype(int)
        print offset_array
        
        for i in range(numpy.shape(lst_array)[0]):
            
            lst_array[i] = numpy.roll(lst_array[i], offset_array[i]) #Bit of a hatchet job...
            


        # Because of the fact HERA only observes for part of the day, we end up with some records
        # eventually "drifting" out of our aligned LST window. As we roll the array to do the
        # alignment we can mask off these loose ends. 
        lst_array = numpy.ma.masked_array(lst_array)
        unaligned_index = numpy.shape(lst_array)[1] + offset_array[0]
        lst_array[:,unaligned_index:] = numpy.ma.masked

        return lst_array

    def align_timestamps(self):
        """
        Generates the closure_dictionary and generates a numpy array of aligned
        LST's, which can be used by the LST_Binner class to extract closures of
        choice across successive days.
        """

        closure_dict = self.__extract_closures()
        #print closure_dict
        lst_ref = self.date_set[0]
        lst_ref = closure_dict[lst_ref] #We interpolate to the LST's from the first day of observations.
        #print lst_ref
        print(len(lst_ref))    
        aligned_lsts = self.__align_closures(lst_ref, closure_dict)

        return aligned_lsts, closure_dict
        

#This parses all of the files and breaks them up into datestamps. Each datestamp is then aligned correctly. 
class Julian_Parse(object):
    """ 
    Class to take a set of filepaths spat out from heracasa and create a dictionary of all the
    files, keyed by their Julian Date.

    Also returns the set of all dates found in the directory.

    Attributes:

    filepaths              [List] All filepaths from the directory ending in .npz. Assumed to be heracasa.
    
    file_dict              [Dictionary] Dictionary of npz files, keyed by their date.

    date_set               [Set] The set of dates.
    
    Member Functions:

    __init__()             Initialises an instance of class Julian_Parse.

    __find_unique_dates()  Finds all unique dates and returns the ordered list of the set of dates.

    __build_closure_dict() Takes the ordered list of the set of dates, and filepaths and sorts 
                           them into file_dict.

    break_up_datestamps()  Creates file_dict.

    return_datestamps()    Returns self.file_dict, self.date_set.

    TODO: Some sort of date ranging so you can control how much data we push through the binner.
    """


    
    # We assume all .npz files have 60 timestamps in them.
    def __init__(self, filepaths, npz_size = 60):

        """
        Initialises the Julian Parse Class which takes a full directory of 
        heracasa .npz files and splits them by date.

        """

        self.filepaths = filepaths
        self.file_dict = None
        self.date_set = None

    # Parses all datestamps and converts to a set (finds unique values)
    def __find_unique_dates(self, filepaths):

        """
        Find unique dates from a directory of heracasa files.

        Inputs:

        filepaths [List] List of all .npz filepaths

        """

        detected_datestamps = []

        for file in self.filepaths:
            detected_datestamps.append(file.split('.')[1])
        detected_datestamps = set(detected_datestamps) # Convert list to set.
        detected_datestamps = sorted(detected_datestamps) # Convert set to ordered list.
        print(detected_datestamps)
        return(detected_datestamps)

    # Takes set of datestamps and filepaths and sorts them into a dict.
    # Dict layout is [Key: datestamp, Data: list of all files in that datestamp]
    def __build_closure_dict(self, datestamps, filepaths):

        """
        Splits the filepaths into a dictionary keyed by their Julian Date

        Inputs:

        datestamps [List] An ordered list of the datestamps.

        filepaths [List] List of al .npz filepaths.

        """
        file_dict = {}

        for datestamp in datestamps:
            file_dict[datestamp] = [] #Empty list.

        for datestamp in datestamps:

            for file in filepaths:
                if file.split('.')[1] == datestamp:
                    file_dict[datestamp].append(file)
            file_dict[datestamp] = sorted(file_dict[datestamp]) #Holy shit this just works? How awesome is that.
        #print(file_dict)
        return(file_dict)
        
    # This builds a dictionary of all unique datestamps and their .npz files.
    def break_up_datestamps(self):

        """
        Breaks up the directory of .npz files into a dictionary, keyed by date.

        """
               
        detected_datestamps = self.__find_unique_dates(self.filepaths)
        file_dict = self.__build_closure_dict(detected_datestamps, self.filepaths)
        self.file_dict = file_dict
        self.date_set = detected_datestamps

    # Returns dicionary.
    def return_datestamps(self):

        """
        Returns file_dict and date_set

        """
        return self.file_dict, self.date_set
    


def main():


    # Parse directory for input/output.
    command = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent('''
    ---------------------------------------------------------------------------------------
    HERA LST Aligner and Binner

    Author: James Kent
    Institution: University of Cambridge, 2018
    Email: jck42@cam.ac.uk

    Takes a directory of heracasa created closure phases (best created using recipe), 
    and concatenates all of the .npz files together, aligns the sidereal times, and 
    outputs and LST range of your choice.

    Can also optinally average closures across LSTs as well as calculate the standard
    deviations, which can be helpful in telling you if you there are days/times/triads
    that are not behaving as wanted.
'''))
    command.add_argument('filepath', help = ('.npz files generated from miriad files and ran through heracasa'))
    command.add_argument('working_directory', help = ('where aligned .npz files will be put.'))
    command.add_argument('--average', action='store_true',required=False, 
                         help='Calculate average across LSTs.')
    command.add_argument('--std_dev', action='store_true',required=False, 
                         help='Calculate standard deviations across LSTs.')
    command.add_argument('--lst_start', required=True, metavar='S', type = int,
                         help='Start index of LSTs, bit of trial and error to get it right..')
    command.add_argument('--lst_range', required=True, metavar='R', type = int,
                         help='Range of LSTs (number of columns) to bin.')
    args = command.parse_args()

    if (os.path.exists(args.working_directory) == False):
        os.makedirs(args.working_directory)
        
    # Find all .npz files. Assume they are all from heracasa...
    files = []
    for file in os.listdir(args.filepath):

        if file.endswith(".npz"):
            files.append(file)

    #for file in files:
    #    print(file.split('.')[1])

    # Instantiate Julian_Parse class, and get files/dates
    parser = Julian_Parse(files)
    parser.break_up_datestamps()
    files, dates = parser.return_datestamps()
    # Instantiate LST_alignment class and align timestamps.
    aligner = LST_Alignment(args.filepath,dates,files)
    aligned_lsts, closures = aligner.align_timestamps()
        
    # Instantiate LST_binner class class, then bin LSTS and save to file.
    binner = LST_Binner(aligned_lsts, closures, lst_start=args.lst_start, lst_range=args.lst_range)
    binner.bin_lsts(average=args.average,std_dev=args.std_dev)
    binner.save_binned_lsts(args.working_directory+"output_bin.npz")
    

if __name__=="__main__":
    main()

