# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.1.0-ALPHA
#        DATA LAYER OBJECTS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
import random
import TALOS.Constants as tcc
from TALOS.FileSystem import Storage, DataSetFolder
#from xml.dom import VALIDATION_ERR
#from TALOS.Constants import DS_TRAINING, DS_VALIDATION
from TALOS.DataIterator import MLDataIterator
from TALOS.Core import ScreenLogger




    
#==================================================================================================
class MLDataLayerConst(object):
    CLASSES_DICT_FILENAME           = "classes.dict"
    
    GROUND_TRUTH_INDEX_FILENAME     ="GTSampleSet.index.txt"
    VALIDATION_INDEX_FILENAME       ="VASampleSet.index.txt"
    UNKNOWN_TEST_INEX_FILENAME      ="UTSampleSet.index.txt"

    FILENAME_TEMPLATE_GROUND_TRUTH  ="ground_truth_%03i.pkl"
    FILENAME_TEMPLATE_VALIDATION    ="validation_%03i.pkl"
    FILENAME_TEMPLATE_UNKNOWN_TEST  ="test_%03i.pkl"
    
#==================================================================================================






    
    
        
        
        
        
        
    
#==================================================================================================
class MLSampleSet(object):
    __verboseLevel=1
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_sName=""):
        #........ |  Instance Attributes | ..............................................
        self.Name=p_sName
        self.Parent=p_oParent
        self.Features=None
        self.Targets=None
        self.IDs=None
        
        self.FeaturesTensor=None
        self.TargetsTensor=None
        self.IDsTensor=None
        #................................................................................
    #------------------------------------------------------------------------------------
    @classmethod
    def SetVerboseLevel(cls, p_nLevel):
        cls.__verboseLevel = p_nLevel
    #------------------------------------------------------------------------------------
    def SampleCount(self):
        assert (self.Features.shape[0] == self.Targets.shape[0]), \
                    "uneven features and targets" 
        return self.Targets.shape[0]
    #------------------------------------------------------------------------------------
    def GenerateIDs(self, p_nBaseID=0):
        if self.Features is not None:
            nIDs = np.arange(self.SampleCount(), dtype=np.int32) + p_nBaseID
            self.IDs=nIDs.reshape(self.SampleCount(), 1)
            print("%s: IDs from %i to %i" % (self.Name, self.IDs[0,0], self.IDs[-1,0]))
    #------------------------------------------------------------------------------------
    def Count(self):
        return self.Features.shape[0]
    #------------------------------------------------------------------------------------
    def GetShapesForSampleCount(self, p_nSampleCount=None):
        nFeaturesShape=None
        nTargetsShape=None
        
     
        if self.FeaturesTensor is not None:
            nFeaturesShape = self.FeaturesTensor.get_shape().as_list()
        elif self.Features is not None:
            nFeaturesShape = self.Features.shape
        else:
            raise Exception("No features exist")
          
        if self.TargetsTensor is not None:
            nTargetsShape = self.TargetsTensor.get_shape().as_list()
        elif self.Targets is not None:
            nTargetsShape = self.Targets.shape
        else:
            raise Exception("No features exist")
           
        # If p_nSampleCount remains None this will create a Tensor that can have invariant count of sample
        bIsImageMode=(len(nFeaturesShape) == 4)
        if bIsImageMode:
            # Support for features per square (images)
            nFeaturesShape=[p_nSampleCount, nFeaturesShape[1], nFeaturesShape[2], nFeaturesShape[3]]
        elif len(nFeaturesShape) == 2:
            # Other type of features
            nFeaturesShape=[p_nSampleCount, nFeaturesShape[1]]
        else:
            raise Exception("unsupport feature dimensions")

        nTargetsShape=[p_nSampleCount, nTargetsShape[1]]
            
                        
        if type(self).__verboseLevel >= 1:
            print("|__ Minbatch shapes. Features:", nFeaturesShape, "Targets:", nTargetsShape)
                           
        return nFeaturesShape, nTargetsShape, bIsImageMode         
    #------------------------------------------------------------------------------------
    def GetClassNameBySampleIndex(self, p_nSampleIndex):
        nClassID = int( self.Targets[p_nSampleIndex] ) 
        
        if self.Parent.ClassNames is None:
            sClassName = str(nClassID)
        else:
            assert (nClassID < len(self.Parent.ClassNames)) and (nClassID >= 0), "Unknown class index %i" % nClassID
            sClassName = self.Parent.ClassNames[nClassID]
        
        return sClassName, nClassID
        
    #------------------------------------------------------------------------------------
    def NormalizeFeatures(self):
        if self.Features is not None:
            self.Features[:,:,:,:] = self.Features[:,:,:,:] / 255.0
    #------------------------------------------------------------------------------------
    def GetPack(self, p_nStart, p_nEnd):
        nFeatures = self.Features[p_nStart:p_nEnd,:,:,:] 
        nTargets = self.Targets[p_nStart:p_nEnd,:]  
        nIDs = self.IDs[p_nStart:p_nEnd,:]
        
        sClassNames = []
        for nIndex in range(p_nStart,p_nEnd):
            sClassNames.append(self.GetClassNameBySampleIndex(nIndex))
        sClassNames = np.asarray(sClassNames)
        
        oResult=[]
        oResult.append(nFeatures)
        oResult.append(nTargets)
        oResult.append(nIDs)
        oResult.append(sClassNames)
        
        return oResult
    #------------------------------------------------------------------------------------
#==================================================================================================







#==================================================================================================
class MLDataSetSettings(object):
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........ |  Instance Attributes | ..............................................
        self.RandomSeed=2017
        #................................................................................
    #------------------------------------------------------------------------------------
#==================================================================================================    











    
    
#==================================================================================================
class MLPageFileIterator(object):    
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nPageSize):
        #........ |  Instance Attributes | ..............................................
        self.Parent         = p_oParent
        self.TotalSamples   = self.Parent.TotalSamples
        self.PageSize       = p_nPageSize
        self.Folder         = self.Parent.DataSetFolder.GetPageFolder(self.Parent.CollectionType)
        
        self.PageIndex      = 0
        self.Start          = 0
        self.End            = self.PageSize
        self.EstimatedPages = int(np.ceil(self.TotalSamples / p_nPageSize))
        #................................................................................
    #------------------------------------------------------------------------------------
    def __iter__(self):
        self.PageIndex = 0
        return self
    #------------------------------------------------------------------------------------
    def __next__(self):
        bContinue = self.Start < self.TotalSamples
        if bContinue:
            nIDs = self.Parent.IDs[self.Start:self.End]
            sSampleFiles = self.Parent.SampleFullFileNames[self.Start:self.End]
            nTargets = self.Parent.Targets[self.Start:self.End]            
            sPageFileName = Storage.JoinPath(self.Folder, MLDataIterator.FILENAME_TEMPLATE_PAGE % self.PageIndex)

            self.Start = self.Start + self.PageSize
            self.End = self.Start + self.PageSize
            self.PageIndex += 1
            
            return [sPageFileName, nIDs, sSampleFiles, nTargets]     
        else:
            raise StopIteration
    #------------------------------------------------------------------------------------
#==================================================================================================    










#==================================================================================================
class MLDataSetCollection(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_sCollectionType):
        #........ |  Instance Attributes | ..............................................
        
        # // Aggregated Objects \\
        self.Parent = p_oParent
        self.DataSetFolder = self.Parent.DataSetFolder

        # // Settings and Properties \\
        self.CollectionType = p_sCollectionType
        self.CollectionTypeStr = None
        if p_sCollectionType == tcc.DS_TRAINING:
            self.CollectionTypeStr="train"
        elif p_sCollectionType == tcc.DS_VALIDATION:
            self.CollectionTypeStr="val"
        elif p_sCollectionType == tcc.DS_TESTING:
            self.CollectionTypeStr="test"
        self.MaxSamples=None
        self.TotalSamples=0
        
        # // Composite Objects \\
        self.ClassFolders=None
        self.ClassSamplesFileNames=None
        self.ClassSamplesFullFileNames=None
        self.ClassSamplesAvailable=None
        self.ClassSamplesIDs=None
        
        self.SampleFullFileNames=None
        self.IsStratified = False
        self.IsShuffled = False
        self.IsActive = False
        #................................................................................
    #------------------------------------------------------------------------------------
    def GetIterator(self, p_nFoldNumber, p_nBatchSize):
        sFolder = None
        if self.CollectionType == tcc.DS_TRAINING:
            sFolder = self.DataSetFolder.TrainCacheFolder
        elif self.CollectionType == tcc.DS_VALIDATION:
            sFolder = self.DataSetFolder.ValCacheFolder
        elif self.CollectionType == tcc.DS_TESTING:
            sFolder = self.DataSetFolder.TestCacheFolder
        else:
            raise Exception("Invalid sample set type %d" % self.CollectionType) 
        
        oIterator = MLDataIterator(sFolder, self.TotalSamples, self.Parent.PageSize
                                   , p_bIsValidation=False, p_nFoldNumber=p_nFoldNumber, p_nBatchSize=p_nBatchSize)
        return oIterator        
    #------------------------------------------------------------------------------------
    def LoadSampleIndexFromDisk(self):
        sFileName = self.DataSetFolder.GetSampleIndexFile(self.CollectionType)
        self.Parent.Log.Print("      |__ Searching for sample index:%s" % sFileName)
        
        # Creates/loads the samples for each one of the selected classes    
        bResult = Storage.IsExistingFile(sFileName)
        if bResult:
            oData = Storage.DeserializeObjectFromFile(sFileName)
            self.ClassSamplesFullFileNames = oData["ClassSamplesFullFileNames"]
            self.ClassSamplesFileNames = oData["ClassSamplesFileNames"]
            self.ClassSamplesIDs = oData["ClassSamplesIDs"]
        
        return bResult
    #------------------------------------------------------------------------------------
    def SaveSampleIndexToDisk(self):
        sFileName = self.DataSetFolder.GetSampleIndexFile(self.CollectionType)
        
        oData = {
                   "FormatVersion"               : "TALOS10"
                  ,"ClassSamplesFullFileNames"   : self.ClassSamplesFullFileNames
                  ,"ClassSamplesFileNames"       : self.ClassSamplesFileNames
                  ,"ClassSamplesIDs"             : self.ClassSamplesIDs         
                }
                    
        Storage.SerializeObjectToFile(sFileName, oData, p_bIsOverwritting=True)
    #------------------------------------------------------------------------------------
    def LoadFromDisk(self):
        sFileName = self.DataSetFolder.GetSampleSetFile(self.CollectionType)
        
        
        bResult = Storage.IsExistingFile(sFileName)
        
        if bResult:
            oData = Storage.DeserializeObjectFromFile(sFileName)
            self.IDs = oData["IDs"]
            self.SampleFullFileNames = oData["SampleFullFileNames"]
            self.Targets = oData["Targets"]
            self.IsStratified = oData["IsStratified"]
            self.IsShuffled = oData["IsShuffled"]
            
            self.TotalSamples = len(self.IDs)
            self.Parent.Log.Print("  |__ Total training samples:%d" % self.TotalSamples)
        
        return bResult
    #------------------------------------------------------------------------------------
    def SaveToDisk(self):
        sFileName = self.DataSetFolder.GetSampleSetFile(self.CollectionType)
        
        oData = {  
                   "FormatVersion"          : "TALOS10"
                  ,"TotalSamples"           : self.TotalSamples
                  ,"IDs"                    : self.IDs
                  ,"SampleFullFileNames"    : self.SampleFullFileNames
                  ,"Targets"                : self.Targets
                  ,"IsStratified"           : self.IsStratified
                  ,"IsShuffled"             : self.IsShuffled
                }
        Storage.SerializeObjectToFile(sFileName, oData, p_bIsOverwritting=True)
    #------------------------------------------------------------------------------------
    def GetFileNameByID(self, p_nID):
        sFileName = None
        for nIndex, nID in enumerate(self.IDs):
            if nID == p_nID:
                sFileName = self.SampleFullFileNames[nIndex]
                
        return sFileName  
    #------------------------------------------------------------------------------------        
    def CopySelectedSamplesToFolder(self, p_nSelectedSampleIDs, p_sDestFolder):
        for nID in p_nSelectedSampleIDs:
            sFileName = self.GetFileNameByID(nID)
            print(nID, sFileName)
            Storage.CopyFileToFolder(sFileName, p_sDestFolder)         
    #------------------------------------------------------------------------------------
    def TrimClassSamples(self, p_oSampleFullFilesNameNotTrimmed, p_oSampleFileNamesNotTrimmed):
        nClassCount = len(p_oSampleFullFilesNameNotTrimmed)
        assert nClassCount == len(p_oSampleFileNamesNotTrimmed), "Invalid sample file name collections"
         
         
        nClassCounter=np.zeros((nClassCount), np.int32)
        
        self.ClassSamplesFullFileNames=[]
        self.ClassSamplesFileNames=[]

        for nClassIndex in range(0, nClassCount):
            if self.MaxSamples is None:
                oClassSamplesFullFileNames=p_oSampleFullFilesNameNotTrimmed[nClassIndex]
                oClassSampleFileNames=p_oSampleFileNamesNotTrimmed[nClassIndex]
            else:            
                oClassSamplesFullFileNames=[]
                oClassSampleFileNames=[]        
                
            self.ClassSamplesFullFileNames.append(oClassSamplesFullFileNames)
            self.ClassSamplesFileNames.append(oClassSampleFileNames)                
        
        # Trims the class samples evenly distributed so that the dataset has exactly MaxSamples
        if self.MaxSamples is not None: 
            nCount = 0
            while nCount < self.MaxSamples:
                for nClassIndex, oClassSamplesFullFileNames in enumerate(p_oSampleFullFilesNameNotTrimmed):
                    oClassSampleFileNames=p_oSampleFileNamesNotTrimmed[nClassIndex]
                    
                    nCurrentClassCounter = nClassCounter[nClassIndex]
                    if nCurrentClassCounter < len(oClassSamplesFullFileNames):
                        #self.Log.Print("%d/%d   class:%d current:%d samples:%d" % (nCount+1, self.MaxTrainingSamplesCount, nClassIndex, nCurrentClassCounter, len(oClassSampleFileNames)), )
                        
                        self.ClassSamplesFullFileNames[nClassIndex].append(oClassSamplesFullFileNames[nCurrentClassCounter])
                        self.ClassSamplesFileNames[nClassIndex].append(oClassSampleFileNames[nCurrentClassCounter])
                        nClassCounter[nClassIndex] += 1
                        nCount += 1
        
        # Generates the sample IDs and counts the total samples 
        self.TotalSamples = 0
        self.ClassSamplesIDs=[]  
        nID = 0          
        for nClassIndex, oSamplesOfClass in enumerate(self.ClassSamplesFullFileNames):
            # Builds the IDs for the current class samples
            nSampleIDs = []
            for sSample in oSamplesOfClass:
                nSampleIDs.append(nID)
                nID += 1   
            self.ClassSamplesIDs.append(nSampleIDs)
            #self.Log.Print("Class:%d" % nClassIndex, "Count:%d" % len(oClassSamples))
            self.TotalSamples += len(oSamplesOfClass)
            
        if self.MaxSamples is not None:
            assert self.MaxSamples == self.TotalSamples, "Trimming failes with %d total samples" % self.TotalSamples  
    #------------------------------------------------------------------------------------
    def StratifySamples(self):
        # Stratifies the samples into temporary collections picking one from each class
        nClassCounter=np.zeros((len(self.Parent.ClassCodes)), np.int32)
        nIDs=[]
        sSamples=[]
        nTargets=[]
                
        nSampleID = 0
        while nSampleID < self.TotalSamples:
            for nClassIndex, oClassSamples in enumerate(self.ClassSamplesFullFileNames):
                oClassSampleIDs = self.ClassSamplesIDs[nClassIndex]
                nCurrentClassCounter = nClassCounter[nClassIndex]

                if nCurrentClassCounter < len(oClassSampleIDs):
                    #if type(self).__verboseLevel >=1:
                    #    self.Log.Print("strata", len(oClassSampleIDs), nSampleID, nClassIndex, nCurrentClassCounter)
                    nIDs.append(oClassSampleIDs[nCurrentClassCounter])
                    sSamples.append(oClassSamples[nCurrentClassCounter])
                    nTargets.append(nClassIndex)
                    nClassCounter[nClassIndex] += 1
                    nSampleID += 1
        
        
        self.IDs = nIDs
        self.SampleFullFileNames = sSamples
        self.Targets = nTargets
        self.IsStratified = True
    #------------------------------------------------------------------------------------
    def __shuffleArrays(self, *arrs):
        """ shuffle.
     
        Shuffle given arrays at unison, along first axis.
     
        Arguments:
            *arrs: Each array to shuffle at unison as a parameter.
     
        Returns:
            Tuple of shuffled arrays.
     
        """
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
        p = np.random.permutation(len(arrs[0]))
         
        return tuple(arr[p] for arr in arrs)          
    #------------------------------------------------------------------------------------
    def ShufflePages(self, p_nPageSize):    
        # Keeps the existing sample collections
        nIDs = self.IDs
        sSamples = self.SampleFullFileNames
        nTargets = self.Targets
                    
        # Clears the collections
        self.IDs = []
        self.SampleFullFileNames = []
        self.Targets = []
        
        # Splits samples into pages and shuffles the order of samples in each page        
        nStart = 0
        nEnd = nStart + p_nPageSize
        while nStart <  self.TotalSamples:
            nPageIDs, sPageSamples, nPageTargets = self.__shuffleArrays(nIDs[nStart: nEnd], sSamples[nStart:nEnd], nTargets[nStart:nEnd])

            for nIndex, nID in enumerate(nPageIDs):
                self.IDs.append(nID)
                self.SampleFullFileNames.append(sPageSamples[nIndex])
                self.Targets.append(nPageTargets[nIndex])
                
            nStart = nStart + p_nPageSize
            nEnd = nStart + p_nPageSize      
        
        self.IsShuffled = True        
    #------------------------------------------------------------------------------------
    def PageIterator(self, p_nPageSize):
        return MLPageFileIterator(self, p_nPageSize)
    #------------------------------------------------------------------------------------
    def DumpClassFolders(self):
        # Dumps the class folders to a text file            
        with open(Storage.JoinPath(self.DataSetFolder.DictFolder, "%s-%s.class.folders.txt" % (self.Parent.Name, self.CollectionTypeStr)), "w") as oOutFile:
            for nClassIndex, sClassFolder in enumerate(self.ClassFolders):
                print("[%.4d] %s" % (nClassIndex, sClassFolder), file=oOutFile) 
    #------------------------------------------------------------------------------------            
    def DumpSampleIDs(self):
        # Dumps the available class samples count to a text file
        with open(Storage.JoinPath(self.DataSetFolder.DictFolder, "%s-%s.class.samples.available.txt" % (self.Parent.Name, self.CollectionTypeStr)), "w") as oOutFile:
            for nClassIndex, nCount in enumerate(self.ClassSamplesAvailable):
                sClassFolder = self.ClassFolders[nClassIndex]
                print("[%.4d] %d %s" % (nClassIndex, nCount, sClassFolder), file=oOutFile)
        
        # Creates the class sample IDs from the class samples file list and dumps the class samples to a text file
        with open(Storage.JoinPath(self.DataSetFolder.DictFolder, "%s-%s.samples.txt" % (self.Parent.Name, self.CollectionTypeStr)), "w") as oOutFile:
            for nClassIndex, oSamplesOfClass in enumerate(self.ClassSamplesFullFileNames):
                oSamplesFilesOfClass=self.ClassSamplesFileNames[nClassIndex]
                oSamplesIDs=self.ClassSamplesIDs[nClassIndex]
                for nSampleIndex, sSampleFullFileName in enumerate(oSamplesOfClass):
                    sSampleFileName = oSamplesFilesOfClass[nSampleIndex]
                    nID = oSamplesIDs[nSampleIndex]
                    print("#%.7d [%.4d] %s %s" % (nID, nClassIndex, sSampleFileName.ljust(32), sSampleFullFileName), file=oOutFile)
#     #------------------------------------------------------------------------------------
#     def DumpStratifiedSamples(self, p_nIDs, p_nTargets, p_sSamples):
#         with open(Storage.JoinPath(self.DataSetFolder.DictFolder, "%s-%s.samples.stratified.txt" % (self.Parent.Name, self.CollectionTypeStr)), "w") as oOutFile:
#             for nIndex, nID in enumerate(p_nIDs):
#                 print("#%.7d [%.4d] %s" % (nID, self.Targets[nIndex], self.SampleFiles[nIndex]), file=oOutFile)
    #------------------------------------------------------------------------------------
    def DumpSampleFullFileNamesToDisk(self, p_sSuffix):
        # Dumps the class samples to a text file
        with open(Storage.JoinPath(self.DataSetFolder.DictFolder, "%s-%s.samples.%s.txt" % (self.Parent.Name, self.CollectionTypeStr, p_sSuffix)), "w") as oOutFile:
            for nIndex, nID in enumerate(self.IDs):
                print("#%.7d [%.4d] %s" % (nID, self.Targets[nIndex], self.SampleFullFileNames[nIndex]), file=oOutFile)   
#==================================================================================================



        
        
        
        
                
#==================================================================================================
class MLDataSetBase(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sDataSetFolder, p_nClassCount):
        #........ |  Instance Attributes | ..............................................
        self.ClassCount = p_nClassCount
        self.ClassCodes=None
        self.ClassDescr=None        
        self.DataSetFolder=DataSetFolder(p_sDataSetFolder)
        self.Name = "DS"
        self.Log = ScreenLogger
        self.Train = MLDataSetCollection(self, tcc.DS_TRAINING)
        self.Validation = MLDataSetCollection(self, tcc.DS_VALIDATION)
        self.Testing = MLDataSetCollection(self, tcc.DS_TESTING)
        #................................................................................
    #------------------------------------------------------------------------------------
    def DumpClassesToDisk(self):
        # Dumps the class folders to a text file            
        with open(Storage.JoinPath(self.DataSetFolder.DictFolder, "%s-classes.txt" % (self.Name)), "w") as oOutFile:
            for nClassIndex, sClassCode in enumerate(self.ClassCodes):
                print("[%.4d] %s %s" % (nClassIndex, sClassCode, self.ClassDescr[nClassIndex]), file=oOutFile)         
    #------------------------------------------------------------------------------------
    def PickRandomSamplesFromClassFolder(self, p_sSourceClassFolder, p_nSamplesPerClass):
        """ Selects random samples from the given class folder """
        oFileNamesFull=[]  
        oFileNames=[]
        
        sFileNames = Storage.GetFilesSorted(p_sSourceClassFolder)
        random.shuffle(sFileNames)            
        
        nAvailableSamplesCount = len(sFileNames)
                        
        nSamplesPerClass = 0     
        for sSampleFileName in sFileNames:
            if p_nSamplesPerClass is None:
                bMustPick = True
            else:
                bMustPick = nSamplesPerClass < p_nSamplesPerClass
#                 
#                 sSampleFullFileName = os.path.join(p_sSourceClassFolder, sSampleFileName)
#                 nSamplesPerClass += 1
#                 oFileNamesFull.append(sSampleFullFileName)
#                 oFileNames.append(sSampleFileName)
#             elif nSamplesPerClass < p_nSamplesPerClass:
#                 sSampleFullFileName = os.path.join(p_sSourceClassFolder, sSampleFileName)
#                 nSamplesPerClass += 1
#                 oFileNamesFull.append(sSampleFullFileName)
#                 oFileNames.append(sSampleFileName)
            if bMustPick:   
                sSampleFullFileName = Storage.JoinPath(p_sSourceClassFolder, sSampleFileName)
                nSamplesPerClass += 1
                oFileNamesFull.append(sSampleFullFileName)
                oFileNames.append(sSampleFileName)
                             
        #print(len(oFileNames), len(oFileNamesFull))
        
        return oFileNamesFull, oFileNames, nAvailableSamplesCount
    #------------------------------------------------------------------------------------
    def Create(self, p_oLog=None):
        if p_oLog is not None:
            self.Log = p_oLog
        self.Log.Print("[>] DataSet: %s" % self.Name)            
        self.Initialize()    
    #------------------------------------------------------------------------------------
    def Initialize(self):
        pass
    #------------------------------------------------------------------------------------
#==================================================================================================

    
    
    
    





#==================================================================================================
class MLDataSet(object):
    
    __verboseLevel = 1
    
    BASE_DATASET_FOLDER = "C:\\Image.DataSets\\"
    
    
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_sDataSetName="Test", p_sVariationName="", p_nStartID=0, p_nSwitchToSetType=None, p_bIsLoadingOnce=False):
        #........ |  Instance Attributes | ..............................................
        self.DataSetName=p_sDataSetName
        self.SwitchToSetType=p_nSwitchToSetType        
        self.VariationName=p_sVariationName
        if p_sVariationName == "":
            self.Name=p_sDataSetName
        else:
            self.Name=p_sDataSetName + "_" + p_sVariationName
        


        self.Settings=MLDataSetSettings()

        self.DataSetFolder= MLDataSet.BASE_DATASET_FOLDER + self.Name + "\\"
        self.GroundTruth=MLSampleSet(self, "GTSet")
        self.Validation=MLSampleSet(self, "VASet")
        self.Test=MLSampleSet(self, "UTSet")
        self.QuerySet=MLSampleSet(self, "QuerySet")
        self.UsedSet=self.GroundTruth
        self.ClassNames=None
        self.SelectedClassIDs=[]
        self.Session=None
        self.__isDataSetCreated=False
        self.IsRandomData=False
        self.IsLoadingOnce=p_bIsLoadingOnce
        
        if (p_nStartID is None) and (p_nSwitchToSetType is not None):
            if self.SwitchToSetType == tcc.USE_UNKNOWN_TEST_SET:
                self.StartID=3000
            elif self.SwitchToSetType == tcc.USE_QUERY_TEST_SET:
                self.StartID=10000 
            elif self.SwitchToSetType == tcc.USE_CUSTOM_QUERY_TEST_SET:
                self.StartID=20000
            else:
                self.StartID=0
        else:
            self.StartID=p_nStartID
        self.IsLoaded=False
        #................................................................................
        Storage.EnsurePathExists(self.DataSetFolder)
        print("[>] Using %s dataset" % self.Name)
        
    #------------------------------------------------------------------------------------
    @classmethod
    def SetBaseDataSetFolder(cls, p_sBaseDataSetFolder):
        cls.BASE_DATASET_FOLDER =  p_sBaseDataSetFolder
    #------------------------------------------------------------------------------------
    def CreateDataSet(self):
        if not self.__isDataSetCreated:
            self.DoOnCreateDataSet()
            self.__isDataSetCreated = True
    #------------------------------------------------------------------------------------
    def SwitchToTraining(self):
        self.UsedSet=self.GroundTruth
        print ("      |>  Switching to GT set")
    #------------------------------------------------------------------------------------
    def SwitchToTest(self):
        self.UsedSet=self.Test
        print ("      |>  Switching to UT set")
    #------------------------------------------------------------------------------------
    def SwitchToQueries(self):
        self.UsedSet=self.QuerySet
        print ("      |>  Switching to QT set")
    #------------------------------------------------------------------------------------
    def DoOnCreateDataSet(self):
        pass
    #------------------------------------------------------------------------------------
    def DoOnLoadDataSet(self):
        pass    
    #------------------------------------------------------------------------------------
    def DoOnRunTensors(self):
        #TODO: Preprocessing on Features using tensors
                
        # Emulates loading of data into the main threads memory space
        self.GroundTruth.Features, self.GroundTruth.Targets = self.Session.run(
            [   self.GroundTruth.FeaturesTensor
              , self.GroundTruth.TargetsTensor
             ])
    #------------------------------------------------------------------------------------
    def EnsureDataSpace(self):
        pass
    #------------------------------------------------------------------------------------
    def LoadSamples(self, p_oSession=None):
        if self.IsLoadingOnce and self.IsLoaded:
            print("DataSet %s already loaded." % self.Name) 
            return
        
        
        if not self.__isDataSetCreated:
            self.DoOnCreateDataSet()
            self.__isDataSetCreated = True

        self.Session=p_oSession
        bIsLoaded=False
        if self.IsRandomData:
            # It will run after the session is created
            if self.Session is not None: 
                # Creates random datasets using the tensors, and readjusts them
                self.DoOnRunTensors()
                self.DoOnLoadDataSet()     
                bIsLoaded=True       
        else:
            # It will run before the session is created
            if self.Session is None:
                self.DoOnLoadDataSet()
                bIsLoaded=True
        
        if bIsLoaded:    
            nStartID=self.StartID
            
            # Check if features are loaded (... and so the rest of the data)
            if self.GroundTruth.Features is not None:
                # Generates incremental IDs for the samples of the loaded dataset  
                self.GroundTruth.GenerateIDs(nStartID)
                nStartID=self.GroundTruth.SampleCount()
                
            if self.Test.Features is not None:
                self.Test.GenerateIDs(nStartID)
                nStartID=self.Test.SampleCount()
                
            if self.QuerySet.Features is not None:
                self.QuerySet.GenerateIDs(nStartID)
                
        self.IsLoaded = bIsLoaded
        
        if self.SwitchToSetType is not None:
            if self.SwitchToSetType == tcc.USE_GROUND_TRUTH_SET:
                self.SwitchToTraining()
            elif self.SwitchToSetType == tcc.USE_UNKNOWN_TEST_SET:
                self.SwitchToTest()
            elif self.SwitchToSetType == tcc.USE_QUERY_TEST_SET:
                self.SwitchToQueries()
    #------------------------------------------------------------------------------------
#==================================================================================================








   
#==================================================================================================
#[TO BE REFACTORED]
class MLFoldNavigator(object):
    __verboseLevel=2
    #------------------------------------------------------------------------------------
    def __init__(self, p_nTotalSampleCount, p_nFoldsCount=10, p_nValidationPercentage=0.1,  p_nSubFoldsPerFold=4, p_bIsValidationNavigator=False, p_bIsShuffling=False, p_nBatchSamplesCount=None):
        #........ |  Instance Attributes | ..............................................
        self.IsValidationNavigator=p_bIsValidationNavigator
      
        self.FoldsCount = p_nFoldsCount
        self.TotalSampleCount = p_nTotalSampleCount
        self.ValidationPercentage = p_nValidationPercentage
        self.SubFoldsPerFold = p_nSubFoldsPerFold
        self.IsRecallingAll = (int(p_nValidationPercentage)==1)
        self.IsNavigatingTestSet = (p_nBatchSamplesCount is not None)
        self.BatchSamplesCount = p_nBatchSamplesCount
        

        if self.IsNavigatingTestSet:
            #Custom arithmetic used only for testing
            assert (self.TotalSampleCount / self.BatchSamplesCount) == int(self.TotalSampleCount / self.BatchSamplesCount),\
                    "Non integer subfoldcount using %i samples per batch" % self.BatchSamplesCount
                    
            self.SubFoldsCount = int(self.TotalSampleCount / self.BatchSamplesCount)
            self.ValidationSubFoldsCount = int(self.SubFoldsCount * p_nValidationPercentage)
            self.TrainingSubFoldsCount = self.SubFoldsCount - self.ValidationSubFoldsCount
    
            self.SubFoldSampleCount = self.BatchSamplesCount
            self.ValidationSamplesCount = self.ValidationSubFoldsCount * self.SubFoldSampleCount 
            self.TrainingSamplesCount = self.TotalSampleCount - self.ValidationSamplesCount        
        else:
            self.SubFoldsCount = self.FoldsCount * self.SubFoldsPerFold
            self.ValidationSubFoldsCount = int(self.SubFoldsCount * p_nValidationPercentage)
            self.TrainingSubFoldsCount = self.SubFoldsCount - self.ValidationSubFoldsCount
    
            self.SubFoldSampleCount = int(self.TotalSampleCount/self.SubFoldsCount)
            self.ValidationSamplesCount = self.ValidationSubFoldsCount * self.SubFoldSampleCount 
            self.TrainingSamplesCount = self.TotalSampleCount - self.ValidationSamplesCount 
        
        if self.IsValidationNavigator:
            self.Char="V"
            self.MaxSubFoldsCount=self.ValidationSubFoldsCount
        else:
            self.Char="T"
            self.MaxSubFoldsCount=self.TrainingSubFoldsCount  
                    
        self.Mask=[False]*self.SubFoldsCount
        self.Mask=np.asarray(self.Mask)
        self.SubFoldRanges=[]
        self.FeedIndexes=None
        self.IsShuffling=p_bIsShuffling
        
        
        self.IsLeaveOneOutValidation=True
        
        # ... Control Variables ...
        self.PickedFold=None
        self.RemainingSubFolds=None
        self.Position=self.MaxSubFoldsCount + 1        # Begin with EOF value to force round-robin first
        #...............................................................................
        #Picks the last fold, input parameter to the method is the fold's number not the index
        #self.PickFoldForValidation(self.FoldsCount) 

        
        #V Subfolds Total=40 (samples:2945), Training=0 (samples:25), Validation=40 (samples:2920)
        
        
        
        if type(self).__verboseLevel>=0:
            print("%s Subfolds Total=%i (samples:%i), Training=%i (samples:%i), Validation=%i (samples:%i)" 
                  % (  self.Char 
                     , self.SubFoldsCount, self.TotalSampleCount
                     , self.TrainingSubFoldsCount, self.TrainingSamplesCount
                     , self.ValidationSubFoldsCount, self.ValidationSamplesCount
                     )
                )
        
        self.Restart()
    #------------------------------------------------------------------------------------
    @classmethod
    def SetVerboseLevel(cls, p_nLevel):
        cls.__verboseLevel = p_nLevel         
    #------------------------------------------------------------------------------------
    def __getSampleIDs(self, p_bIsForValidation):
        # Creates the sub fold ranges upon first use
        if self.SubFoldRanges==[]:
            for nIndex in range(self.SubFoldsCount):
                nStart = nIndex*self.SubFoldSampleCount
                nEnd = (nIndex+1)*self.SubFoldSampleCount 
                self.SubFoldRanges.append(slice(nStart, nEnd))

        # Gets the valid subfold indices for training/validation
        nIndexes=[nIndex for nIndex,bIsForValidation in enumerate(self.Mask) if bIsForValidation==p_bIsForValidation]
        
        # Gathers all ids of the training/validation set
        nResult = None
        for nIndex in nIndexes:
            a = np.arange(self.SubFoldRanges[nIndex].start, self.SubFoldRanges[nIndex].stop)
            if nResult is None:
                nResult = a
            else:
                nResult = np.concatenate((nResult, a), axis=0)
        
        nResult = nResult.reshape((nResult.shape[0], 1))
        
        return nResult
    #------------------------------------------------------------------------------------
    def GetTrainingSampleIDs(self):
        return self.__getSampleIDs(False)
    #------------------------------------------------------------------------------------
    def GetValidationSampleIDs(self):
        return self.__getSampleIDs(True)        
    #------------------------------------------------------------------------------------
    def PickFoldForValidation(self, p_nValidationFoldNumber):
        self.PickedFold = p_nValidationFoldNumber - 1
        
        if self.IsRecallingAll:
            # All subfolds are for recall, nothing is for training
            self.Mask[:]=True 
        else:
            # Subfolds are not for validation (default)
            self.Mask[:]=False
        
            nStart = self.PickedFold*self.SubFoldsPerFold
            nEnd = self.PickedFold*self.SubFoldsPerFold + self.SubFoldsPerFold
            if type(self).__verboseLevel>=1:
                print("Cross-Validation: %s Picked flagging out fold %i for validation, subfolds %i to %i" % (self.Char, self.PickedFold, nStart, nEnd-1))
    
            # Flags subfolds of validation with True
            self.Mask[nStart:nEnd]=True
        if type(self).__verboseLevel>=2:
            print(self.Mask)
    #------------------------------------------------------------------------------------
    def Restart(self):
        if self.IsValidationNavigator:
            self.RemainingSubFolds=self.ValidationSubFoldsCount
        else:
            self.RemainingSubFolds=self.TrainingSubFoldsCount
        if (self).__verboseLevel >= 1:
            print("\n\t\t\t|>   %s Restarted " % self.Char)        
    #------------------------------------------------------------------------------------
    def PopNextSubFold(self, p_bIsRoundRobin=False):
        nIndex = -1
        if not p_bIsRoundRobin:            
            if self.RemainingSubFolds == 0:
                nIndex = None
                
        if nIndex is not None:
            if self.EndOfSubFolds():
                nIndex = self.FirstSubFold()
            else:
                nIndex = self.NextSubFold()

            if not p_bIsRoundRobin:
                self.RemainingSubFolds -= 1
                            
                if type(self).__verboseLevel>=2:
                    print("\t\t\t(->) Remaining %s subfolds %i" % (self.Char, self.RemainingSubFolds) )                
                         
        return nIndex, self.RemainingSubFolds
    #-----------------------------------------------------------------------------------
    def GetFeedIndexing(self):
        # Gets all the indexes:
        #    For training are flagged with False
        #    For validation/recall are flagged with True
        nIndexes=[nIndex for nIndex,bIsForValidation in enumerate(self.Mask) if bIsForValidation==self.IsValidationNavigator]
        
        # Shuffles the indexes that will be returned
        if self.IsValidationNavigator:
            # Recall indexes have no shuffling
            if (self).__verboseLevel == 1:
                print("\n\t\t\t|>>> R Indexes:%i\n" % len(nIndexes))
        else:
            if self.IsShuffling:
                np.random.shuffle(nIndexes)
                if (self).__verboseLevel == 1:
                    print("\n\t\t\t|>>> T Indexes shuffled:%i\n" % len(nIndexes))
            else:
                if (self).__verboseLevel == 1:
                    print("\n\t\t\t|>>> T Indexes:%i\n" % len(nIndexes))
            

        if type(self).__verboseLevel >= 2:
            print("........ %s ......." % self.Char)
            print(nIndexes)
            print("...................")
                        
        return nIndexes
    #-----------------------------------------------------------------------------------
    def FirstSubFold(self):
        if not self.IsLeaveOneOutValidation:
            self.PickedFold += 1
            if self.PickedFold == self.FoldsCount:
                self.PickedFold = 0
            self.PickFoldForValidation(self.PickedFold)
            
        self.FeedIndexes = self.GetFeedIndexing()
        
        self.Position=0
        return self.NextSubFold()
    #------------------------------------------------------------------------------------
    def NextSubFold(self):
        nResult = self.FeedIndexes[self.Position]
        self.Position += 1
        return nResult    
    #------------------------------------------------------------------------------------
    def EndOfSubFolds(self):
        return self.Position >= self.MaxSubFoldsCount
    #------------------------------------------------------------------------------------

#==================================================================================================
#[TO BE REFACTORED]
class MLSamples(object):
    
    __verboseLevel = 1
    
    BASE_DATASET_FOLDER = "I:\\DataSets\\"
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_sDataSetName="Test", p_sVariationName="", p_nStartID=0, p_nSwitchToSetType=None, p_bIsLoadingOnce=False):
        #........ |  Instance Attributes | ..............................................
        self.DataSetName=p_sDataSetName
        self.SwitchToSetType=p_nSwitchToSetType        
        self.VariationName=p_sVariationName
        if p_sVariationName == "":
            self.Name=p_sDataSetName
        else:
            self.Name=p_sDataSetName + "_" + p_sVariationName
        self.DataSetFolder= type(self).BASE_DATASET_FOLDER + self.Name + "\\"
        self.GroundTruth=MLSet(self, "GTSet")
        self.Test=MLSet(self, "UTSet")
        self.QuerySet=MLSet(self, "QuerySet")
        self.UsedSet=self.GroundTruth
        self.ClassNames=None
        self.Session=None
        self.__isDataSetCreated=False
        self.IsRandomData=False
        self.IsLoadingOnce=p_bIsLoadingOnce
        
        if (p_nStartID is None) and (p_nSwitchToSetType is not None):
            if self.SwitchToSetType == tcc.USE_UNKNOWN_TEST_SET:
                self.StartID=3000
            elif self.SwitchToSetType == tcc.USE_QUERY_TEST_SET:
                self.StartID=10000 
            elif self.SwitchToSetType == tcc.USE_CUSTOM_QUERY_TEST_SET:
                self.StartID=20000
            else:
                self.StartID=0
        else:
            self.StartID=p_nStartID
        self.IsLoaded=False
        #................................................................................
        Storage.EnsurePathExists(self.DataSetFolder)
        print("[>] Using %s dataset" % self.Name)
        
    #------------------------------------------------------------------------------------
    @classmethod
    def SetBaseDataSetFolder(cls, p_sBaseDataSetFolder):
        cls.BASE_DATASET_FOLDER =  p_sBaseDataSetFolder
    #------------------------------------------------------------------------------------
    def CreateDataSet(self):
        if not self.__isDataSetCreated:
            self.DoOnCreateDataSet()
            self.__isDataSetCreated = True
    #------------------------------------------------------------------------------------
    def SwitchToTraining(self):
        self.UsedSet=self.GroundTruth
        print ("      |>  Switching to GT set")
    #------------------------------------------------------------------------------------
    def SwitchToTest(self):
        self.UsedSet=self.Test
        print ("      |>  Switching to UT set")
    #------------------------------------------------------------------------------------
    def SwitchToQueries(self):
        self.UsedSet=self.QuerySet
        print ("      |>  Switching to QT set")
    #------------------------------------------------------------------------------------
    def DoOnCreateDataSet(self):
        pass
    #------------------------------------------------------------------------------------
    def DoOnLoadDataSet(self):
        pass    
    #------------------------------------------------------------------------------------
    def DoOnRunTensors(self):
        #TODO: Preprocessing on Features using tensors
                
        # Emulates loading of data into the main threads memory space
        self.GroundTruth.Features, self.GroundTruth.Targets = self.Session.run(
            [   self.GroundTruth.FeaturesTensor
              , self.GroundTruth.TargetsTensor
             ])
    #------------------------------------------------------------------------------------
    def EnsureDataSpace(self):
        pass
    #------------------------------------------------------------------------------------
    def LoadSamples(self, p_oSession=None):
        if self.IsLoadingOnce and self.IsLoaded:
            print("DataSet %s already loaded." % self.Name) 
            return
        
        
        if not self.__isDataSetCreated:
            self.DoOnCreateDataSet()
            self.__isDataSetCreated = True

        self.Session=p_oSession
        bIsLoaded=False
        if self.IsRandomData:
            # It will run after the session is created
            if self.Session is not None: 
                # Creates random datasets using the tensors, and readjusts them
                self.DoOnRunTensors()
                self.DoOnLoadDataSet()     
                bIsLoaded=True       
        else:
            # It will run before the session is created
            if self.Session is None:
                self.DoOnLoadDataSet()
                bIsLoaded=True
        
        if bIsLoaded:    
            nStartID=self.StartID
            
            # Check if features are loaded (... and so the rest of the data)
            if self.GroundTruth.Features is not None:
                # Generates incremental IDs for the samples of the loaded dataset  
                self.GroundTruth.GenerateIDs(nStartID)
                nStartID=self.GroundTruth.SampleCount()
                
            if self.Test.Features is not None:
                self.Test.GenerateIDs(nStartID)
                nStartID=self.Test.SampleCount()
                
            if self.QuerySet.Features is not None:
                self.QuerySet.GenerateIDs(nStartID)
                
        self.IsLoaded = bIsLoaded
        
        if self.SwitchToSetType is not None:
            if self.SwitchToSetType == tcc.USE_GROUND_TRUTH_SET:
                self.SwitchToTraining()
            elif self.SwitchToSetType == tcc.USE_UNKNOWN_TEST_SET:
                self.SwitchToTest()
            elif self.SwitchToSetType == tcc.USE_QUERY_TEST_SET:
                self.SwitchToQueries()
    #------------------------------------------------------------------------------------
#==================================================================================================




    
    
    
    
    
    
    
    
    
    











#==================================================================================================
#[TO BE REFACTORED]
class MLSet(object):
    __verboseLevel=1
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_sName=""):
        #........ |  Instance Attributes | ..............................................
        self.Name=p_sName
        self.Parent=p_oParent
        self.Features=None
        self.Targets=None
        self.IDs=None
        
        self.FeaturesTensor=None
        self.TargetsTensor=None
        self.IDsTensor=None
        #................................................................................
    #------------------------------------------------------------------------------------
    @classmethod
    def SetVerboseLevel(cls, p_nLevel):
        cls.__verboseLevel = p_nLevel
    #------------------------------------------------------------------------------------
    def SampleCount(self):
        assert (self.Features.shape[0] == self.Targets.shape[0]), \
                    "uneven features and targets" 
        return self.Targets.shape[0]
    #------------------------------------------------------------------------------------
    def GenerateIDs(self, p_nBaseID=0):
        if self.Features is not None:
            nIDs = np.arange(self.SampleCount(), dtype=np.int32) + p_nBaseID
            self.IDs=nIDs.reshape(self.SampleCount(), 1)
            print("%s: IDs from %i to %i" % (self.Name, self.IDs[0,0], self.IDs[-1,0]))
    #------------------------------------------------------------------------------------
    def Count(self):
        return self.Features.shape[0]
    #------------------------------------------------------------------------------------
    def GetShapesForSampleCount(self, p_nSampleCount=None):
        nFeaturesShape=None
        nTargetsShape=None
        
     
        if self.FeaturesTensor is not None:
            nFeaturesShape = self.FeaturesTensor.get_shape().as_list()
        elif self.Features is not None:
            nFeaturesShape = self.Features.shape
        else:
            raise Exception("No features exist")
          
        if self.TargetsTensor is not None:
            nTargetsShape = self.TargetsTensor.get_shape().as_list()
        elif self.Targets is not None:
            nTargetsShape = self.Targets.shape
        else:
            raise Exception("No features exist")
           
        # If p_nSampleCount remains None this will create a Tensor that can have invariant count of sample
        bIsImageMode=(len(nFeaturesShape) == 4)
        if bIsImageMode:
            # Support for features per square (images)
            nFeaturesShape=[p_nSampleCount, nFeaturesShape[1], nFeaturesShape[2], nFeaturesShape[3]]
        elif len(nFeaturesShape) == 2:
            # Other type of features
            nFeaturesShape=[p_nSampleCount, nFeaturesShape[1]]
        else:
            raise Exception("unsupport feature dimensions")

        nTargetsShape=[p_nSampleCount, nTargetsShape[1]]
            
                        
        if type(self).__verboseLevel >= 1:
            print("|__ Minbatch shapes. Features:", nFeaturesShape, "Targets:", nTargetsShape)
                           
        return nFeaturesShape, nTargetsShape, bIsImageMode         
    #------------------------------------------------------------------------------------
    def GetClassNameBySampleIndex(self, p_nSampleIndex):
        nClassID = int( self.Targets[p_nSampleIndex] ) 
        
        if self.Parent.ClassNames is None:
            sClassName = str(nClassID)
        else:
            assert (nClassID < len(self.Parent.ClassNames)) and (nClassID >= 0), "Unknown class index %i" % nClassID
            sClassName = self.Parent.ClassNames[nClassID]
        
        return sClassName, nClassID
        
    #------------------------------------------------------------------------------------
    def NormalizeFeatures(self):
        if self.Features is not None:
            self.Features[:,:,:,:] = self.Features[:,:,:,:] / 255.0
    #------------------------------------------------------------------------------------
    def GetPack(self, p_nStart, p_nEnd):
        nFeatures = self.Features[p_nStart:p_nEnd,:,:,:] 
        nTargets = self.Targets[p_nStart:p_nEnd,:]  
        nIDs = self.IDs[p_nStart:p_nEnd,:]
        
        sClassNames = []
        for nIndex in range(p_nStart,p_nEnd):
            sClassNames.append(self.GetClassNameBySampleIndex(nIndex))
        sClassNames = np.asarray(sClassNames)
        
        oResult=[]
        oResult.append(nFeatures)
        oResult.append(nTargets)
        oResult.append(nIDs)
        oResult.append(sClassNames)
        
        return oResult
        
    #------------------------------------------------------------------------------------
#==================================================================================================


