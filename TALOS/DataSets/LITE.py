# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        LITE (LESS IMAGENET TRAINING EXAMPLES) DATASET
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
import random
import TALOS.Images as timg
import TALOS.Constants as tcc
from TALOS.DataLayer import MLDataSetBase#, MLDataLayerConst
from TALOS.FileSystem import Storage
from TALOS.Core import Logger
# from TALOS.DataIterator import MLDataIterator




#==================================================================================================
class ImageDataSetLITE(MLDataSetBase):
    __verboseLevel = 1
    #------------------------------------------------------------------------------------
    def __init__(self, p_sDataSetFolder, p_sClassCount=None):
        super(ImageDataSetLITE, self).__init__(p_sDataSetFolder, p_sClassCount)
        #........ |  Instance Attributes | ..............................................
        self.Log = Logger()
        self.Name = "LITE%d" % self.ClassCount

        self.ImageNetSynSetDict = dict()
        
        self.CaltechClassDescr=None
        self.ImageNetClassID=None
        self.ImageNetClassCodes=[]
        self.ImageNetClassDescr=None
        
        
        self.RandomSeed=2017

        self.IDs=None
        self.SampleFiles=None
        self.Targets=None
        
        self.TrainSamplesPerClass=None
        self.PageSize=None
        
        self.Train.MaxSamples=None
        self.Validation.MaxSamples=None
        self.Testing.MaxSamples=None
        #................................................................................

        
    #------------------------------------------------------------------------------------
    def __determineLITEParameters(self):
        assert self.ClassCount is not None, "Must specify a class count for building the dataset"
        assert self.ClassCount in [20, 30, 50, 100, 200, 250], "Invalid class count %d for LITE dataset generation" % self.ClassCount

        self.PageSize = 480
        if self.ClassCount == 20:
            self.TrainSamplesPerClass = 480  #Total:9600
        elif self.ClassCount == 30:
            self.TrainSamplesPerClass = 640  #Total:19200 
        elif self.ClassCount == 50:
            self.TrainSamplesPerClass = 768  #Total:38400
        elif self.ClassCount == 100:
            self.TrainSamplesPerClass = 768  #Total:76800
        elif self.ClassCount == 200:
            self.TrainSamplesPerClass = 720  #Total:144000
        elif self.ClassCount == 250:
            self.TrainSamplesPerClass = 960  #Total:240000
        else:
            raise Exception("LITE%d supported" % self.ClassCount)
        #................................................................................        
    #------------------------------------------------------------------------------------
    def __loadImageNetClasses(self):
        print(self.DataSetFolder.ClassNamesFile)
        
        self.ImageNetClassCodes=[]
        
        nCount = 0
        with open(self.DataSetFolder.ClassNamesFile, "r") as oFile:
            for sLine in oFile:
                sCode = sLine[0:10].strip()
                self.ImageNetSynSetDict[sCode] = sLine[10:].strip()
                self.ImageNetClassCodes.append(sCode)
                nCount += 1         
                
                if type(self).__verboseLevel >=1:
                    print(nCount, sCode, self.ImageNetSynSetDict[sCode])
    #------------------------------------------------------------------------------------
    def __determineCommonCaltechClassesForLITE(self):            
        sCaltechClasses=[]
        with open(Storage.JoinPath(self.DataSetFolder.SourceFolder, "caltech101-classes.txt"), "r") as oFile:
            for sLine in oFile:
                sCaltechClasses.append(sLine.strip())
                
        if type(self).__verboseLevel >=1:
            print("Caltech classes: %d" % len(sCaltechClasses))

        self.ClassCodes=[]
        for nIndex, sClassCode in enumerate(self.ImageNetClassCodes):     
            sClassDescriptions = self.ImageNetSynSetDict[sClassCode]
            bFound = any(sClass in sClassDescriptions for sClass in sCaltechClasses)
                
            if bFound:
                sDescriptions=sClassDescriptions.split(",")
                for sClass in sCaltechClasses:
                    sFound=[[sDescr] for sDescr in sDescriptions if sClass==sDescr.strip()]
                    
                    if len(sFound) != 0:
                        self.ClassCodes.append(sClassCode)
                        self.ClassDescr.append(sClass)
                        self.CaltechClassDescr.append(sClass)
                        self.ImageNetClassID.append(nIndex+1)
                        self.ImageNetClassDescr.append(sClassDescriptions)
    #------------------------------------------------------------------------------------
    def __addRandomImageNetClasses(self):
        nAnimalsRange=[0,398]
        nHigh = int(np.floor( (len(self.ImageNetClassCodes)-nAnimalsRange[1]) / (self.ClassCount-len(self.ClassCodes))))
         
        nIndex = 0
        while len(self.ClassCodes) < self.ClassCount:
            sClassCode = self.ImageNetClassCodes[nIndex]
            nRandom = np.random.randint(low=1, high=nHigh)
            if (nRandom==1) and (nIndex > nAnimalsRange[1]) and ( len(self.ClassCodes) < self.ClassCount ):
                if not (sClassCode in self.ClassCodes): 
                    sClassDescriptions = self.ImageNetSynSetDict[sClassCode]
                    self.ClassCodes.append(sClassCode)
                    self.ClassDescr.append(self.ImageNetSynSetDict[sClassCode].split(",")[0].strip())
                    self.CaltechClassDescr.append(None)
                    self.ImageNetClassID.append(nIndex+1)
                    self.ImageNetClassDescr.append(sClassDescriptions)
                    
            nIndex += 1
            if nIndex >= len(self.ImageNetClassCodes):
                nIndex = 0                        
    #------------------------------------------------------------------------------------
    def DoSelectClasses(self):
        self.ClassCodes=[]
        self.ClassDescr=[]
        self.CaltechClassDescr=[]
        self.ImageNetClassID=[]
        self.ImageNetClassDescr=[]

        # Loads the codes and descriptions for the ILSVRC2012 synsets of 1000 classes
        self.__loadImageNetClasses()
            
        # Determines the 30 common classes between Caltech101 and ILSVRC2012
        if self.ClassCount <= 250:                
            self.__determineCommonCaltechClassesForLITE()
            # Adds extra random classes from ILSVRC2012
            if self.ClassCount == 20:
                self.ClassCodes = self.ClassCodes[9:29]
                self.ClassDescr = self.ClassDescr[9:29]
                self.CaltechClassDescr = self.CaltechClassDescr[9:29]
                self.ImageNetClassID  = self.ImageNetClassID[9:29]
                self.ImageNetClassDescr   = self.ImageNetClassDescr[9:29]
            elif self.ClassCount == 30:
                pass                
            elif self.ClassCount > 30:
                self.__addRandomImageNetClasses()
            else:
                raise Exception("Class count %d not supported for LITE dataset generation" % self.ClassCount)
        else:
            raise Exception("Class count %d not supported for LITE dataset generation" % self.ClassCount)
    #------------------------------------------------------------------------------------
    def DoPickSamples(self, p_nType=tcc.DS_TRAINING):
        if p_nType == tcc.DS_TRAINING:
            oSet = self.Train
        elif p_nType == tcc.DS_VALIDATION:
            oSet = self.Validation
        elif p_nType == tcc.DS_TESTING:
            oSet = self.Testing
        else:
            raise Exception("Invalid sample set type %d" % p_nType)           
        
        # Decompresses the class folders for the selected classes, creates/loads the class folders collection
        bContinue = True
        if oSet.ClassFolders is None:
            oSet.ClassFolders=[]
            for nClassIndex, sClassCode in enumerate(self.ClassCodes):
                print("%d/%d " % (nClassIndex + 1, len(self.ClassCodes)))
                sSourceClassFolder = self.DataSetFolder.GetClassFolder(sClassCode, p_nType)

                if not Storage.IsExistingPath(sSourceClassFolder):
                    #TODO: Decompression
                    #bContinue=False
                    if p_nType == tcc.DS_TRAINING:
                        sSourceClassFolder = self.DataSetFolder.DecompressSamples(sClassCode)
                    elif p_nType == tcc.DS_VALIDATION:
                        nImageNetClassIndex = None
                        for nSubIndex, sImageNetCode in enumerate(self.ImageNetClassCodes):
                            if sImageNetCode == sClassCode:
                                nImageNetClassIndex = nSubIndex
                                break
                        assert nImageNetClassIndex != None   
                        sSourceClassFolder = self.DataSetFolder.CollectValidationSamples(nImageNetClassIndex, sClassCode)
                oSet.ClassFolders.append(sSourceClassFolder)
        
            oSet.IsActive = bContinue
            
        # Uses the saved flag value
        bContinue = oSet.IsActive
        
        if bContinue: 
            oSet.IsActive = bContinue
            # Dumps the class folders to a text file        
            oSet.DumpClassFolders()
                
            # Creates/loads the samples for each one of the selected classes    
            if not oSet.LoadSampleIndexFromDisk():
                oSet.ClassSamplesAvailable=[]            
                oClassSampleFullFileNamesNotTrimmed=[]
                oClassSampleFileNamesNotTrimmed=[]
                for sClassFolder in oSet.ClassFolders:
                    oSampleFullFilenames, oSampleFileNames,  nAvailableSamples = self.PickRandomSamplesFromClassFolder(sClassFolder, self.TrainSamplesPerClass)    
                    oSet.ClassSamplesAvailable.append(nAvailableSamples)
                    oClassSampleFullFileNamesNotTrimmed.append(oSampleFullFilenames)
                    oClassSampleFileNamesNotTrimmed.append(oSampleFileNames)
                
                oSet.TrimClassSamples(oClassSampleFullFileNamesNotTrimmed, oClassSampleFileNamesNotTrimmed)
                                
                self.__saveClassesToDisk()
                oSet.SaveSampleIndexToDisk()
                
            # Dumps the sample IDs to a text file
            oSet.DumpSampleIDs()
    
            # Creates/loads the samples for each one of the selected classes
            if not oSet.LoadFromDisk():
                oSet.StratifySamples()
                # The stratified order occurs only before shuffling, hence not dumped at each run
                oSet.DumpSampleFullFileNamesToDisk("stratified")
                oSet.SaveToDisk()
            
            # Shuffled the samples if not already shuffled     
            if not oSet.IsShuffled:            
                oSet.ShufflePages(self.PageSize) 
                oSet.SaveToDisk()
                
            oSet.DumpSampleFullFileNamesToDisk("shuffled")
    #------------------------------------------------------------------------------------
    def __loadClassesFromDisk(self):
        bResult = Storage.IsExistingFile(self.DataSetFolder.ClassesFile)
        if bResult:
            oData = Storage.DeserializeObjectFromFile(self.DataSetFolder.ClassesFile)
            
            self.ClassCodes = oData["ClassCodes"]
            self.ClassDescr = oData["ClassDescr"]
            self.ClassCount = len(self.ClassCodes)            
            assert len(self.ClassDescr) == self.ClassCount, "incorrect count of class descriptions %d" % len(self.ClassDescr)
            
            self.Train.ClassFolders  = oData["ClassFoldersTrain"]
            self.Validation.ClassFolders = oData["ClassFoldersVal"]
            self.Testing.ClassFolders = oData["ClassFoldersTest"]

            self.Train.ClassSamplesAvailable = oData["ClassSamplesAvailableTrain"]
            self.Validation.ClassSamplesAvailable = oData["ClassSamplesAvailableVal"]
            self.Testing.ClassSamplesAvailable = oData["ClassSamplesAvailableTest"]
            
            self.Train.IsActive = oData["HasTrain"]
            self.Validation.IsActive = oData["HasVal"]
            self.Testing.IsActive = oData["HasTest"]
            
            self.CaltechClassDescr=oData["CaltechClassDescr"]
            self.ImageNetClassID=oData["ImageNetClassID"]
            self.ImageNetClassCodes=oData["ImageNetClassCodes"]
            self.ImageNetClassDescr=oData["ImageNetClassDescr"]
            
            self.TrainSamplesPerClass=oData["TrainSamplesPerClass"]
            self.PageSize=oData["PageSize"]
                    
            self.Log.Print("  |__ Classes: %d" % self.ClassCount)
        else:
            raise Exception("No dataset found under %s" % self.DataSetFolder.BaseFolder)
             
        return bResult
    #------------------------------------------------------------------------------------
    def __saveClassesToDisk(self):
        oData = { 
                  "FormatVersion"                : "TALOS10"
                 ,"ClassCodes"                   : self.ClassCodes
                 ,"ClassDescr"                   : self.ClassDescr
                 
                 ,"ClassFoldersTrain"            : self.Train.ClassFolders
                 ,"ClassFoldersVal"              : self.Validation.ClassFolders
                 ,"ClassFoldersTest"             : self.Testing.ClassFolders
                 
                 ,"ClassSamplesAvailableTrain"   : self.Train.ClassSamplesAvailable
                 ,"ClassSamplesAvailableVal"     : self.Validation.ClassSamplesAvailable
                 ,"ClassSamplesAvailableTest"    : self.Testing.ClassSamplesAvailable
                 
                 ,"HasTrain"                     : self.Train.IsActive
                 ,"HasVal"                       : self.Validation.IsActive
                 ,"HasTest"                      : self.Testing.IsActive
                 
                 ,"CaltechClassDescr"            : self.CaltechClassDescr 
                 ,"ImageNetClassID"              : self.ImageNetClassID
                 ,"ImageNetClassCodes"           : self.ImageNetClassCodes
                 ,"ImageNetClassDescr"           : self.ImageNetClassDescr
                 
                 ,"TrainSamplesPerClass"         : self.TrainSamplesPerClass
                 ,"PageSize"                     : self.PageSize
                }
        Storage.SerializeObjectToFile(self.DataSetFolder.ClassesFile, oData, p_bIsOverwritting=True)
    #------------------------------------------------------------------------------------
    def Build(self):
        self.__determineLITEParameters()
        
        self.Log.Open(self.DataSetFolder.BaseFolder, p_nCustomLogFileName="Build_Log.txt", p_nLogType=Logger.LOG_TYPE_PROCESS)
        self.Log.WriteLine("Creating dataset ...")
        self.Log.Flush()

        random.seed(self.RandomSeed)
        np.random.seed(self.RandomSeed)

        # Creates/loads the class selection
        if not self.__loadClassesFromDisk():
            self.DoSelectClasses()
            self.__saveClassesToDisk()
        self.DumpClassesToDisk()
              
        self.DoPickSamples(tcc.DS_TRAINING)
        self.DoPickSamples(tcc.DS_VALIDATION)
        #self.DoPickSamples(tcc.DS_TESTING)
        
        self.Log.Flush()
    #------------------------------------------------------------------------------------
    def Save(self, p_nImageDimensions):
        oTrain = self.Train.PageIterator(self.PageSize)
        
        for nPageIndex,oPage in enumerate(oTrain):
            # [sPageFileName, nIDs, sSampleFiles, nTargets]
            sPageFileName = oPage[0]
            nIDs = oPage[1] 
            sSampleFiles = oPage[2]
            nTargets = oPage[3]
            
            nSamples = np.zeros( (len(sSampleFiles),p_nImageDimensions[0],p_nImageDimensions[1],3), dtype=np.uint8)
            print("%d/%d samples:" % (nPageIndex+1, oTrain.EstimatedPages), nSamples.shape)
            
            if not Storage.IsExistingFile(sPageFileName):
                for nIndex, sFileName in enumerate(sSampleFiles):
                    #img = timg.LoadImageAndCropToSize(sFileName, p_tSize=p_nImageDimensions)
                    #nSamples[nIndex,:,:,:]=img[:,:,:]
                    img = timg.LoadImageAndMakeAugmentedSquare(sFileName, p_tSize=p_nImageDimensions)
                    # Place the RGB properties in the 4th dimension of the tensor in order to be Tensorflow ready
                    nSamples[nIndex,:,:,:]=img[0][:,:,:] 
                      
                
                oData = {    "IDs"       : nIDs
                            ,"Samples"   : nSamples
                            ,"Targets"   : nTargets
                        }
                Storage.SerializeObjectToFile(sPageFileName, oData, p_nExtraLabel="%d/%d" % (nPageIndex+1, oTrain.EstimatedPages) )
            else:
                print("  {%d/%d} Exists %s" % (nPageIndex+1, oTrain.EstimatedPages, sPageFileName) )
                
                
        oVal = self.Validation.PageIterator(self.PageSize)
        
        for nPageIndex,oPage in enumerate(oVal):
            # [sPageFileName, nIDs, sSampleFiles, nTargets]
            sPageFileName = oPage[0]
            nIDs = oPage[1] 
            sSampleFiles = oPage[2]
            nTargets = oPage[3]
            
            nSamples = np.zeros( (len(sSampleFiles),p_nImageDimensions[0],p_nImageDimensions[1],3), dtype=np.uint8)
            print("%d/%d samples:" % (nPageIndex+1, oVal.EstimatedPages), nSamples.shape)
            
            if not Storage.IsExistingFile(sPageFileName):
                for nIndex, sFileName in enumerate(sSampleFiles):
                    img = timg.LoadImageAndMakeAugmentedSquare(sFileName, p_tSize=p_nImageDimensions)
                    nSamples[nIndex,:,:,:]=img[0][:,:,:] 
 
                
                oData = {    "IDs"       : nIDs
                            ,"Samples"   : nSamples
                            ,"Targets"   : nTargets
                        }
                Storage.SerializeObjectToFile(sPageFileName, oData, p_nExtraLabel="%d/%d" % (nPageIndex+1, oVal.EstimatedPages) )
            else:
                print("  {%d/%d} Exists %s" % (nPageIndex+1, oVal.EstimatedPages, sPageFileName) ) 
                
                                
                
        oTrain = self.Testing.PageIterator(self.PageSize)
        
        for nPageIndex,oPage in enumerate(oTrain):
            # [sPageFileName, nIDs, sSampleFiles, nTargets]
            sPageFileName = oPage[0]
            nIDs = oPage[1] 
            sSampleFiles = oPage[2]
            nTargets = oPage[3]
            
            nSamples = np.zeros( (len(sSampleFiles),p_nImageDimensions[0],p_nImageDimensions[1],3), dtype=np.uint8)
            print("%d/%d samples:" % (nPageIndex+1, oTrain.EstimatedPages), nSamples.shape)
            
            if not Storage.IsExistingFile(sPageFileName):
                for nIndex, sFileName in enumerate(sSampleFiles):
                    img = timg.LoadImageAndMakeAugmentedSquare(sFileName, p_tSize=p_nImageDimensions)
                    nSamples[nIndex,:,:,:]=img[0][:,:,:] 
                
                oData = {    "IDs"       : nIDs
                            ,"Samples"   : nSamples
                            ,"Targets"   : nTargets
                        }
                Storage.SerializeObjectToFile(sPageFileName, oData, p_nExtraLabel="%d/%d" % (nPageIndex+1, oTrain.EstimatedPages) )
            else:
                print("  {%d/%d} Exists %s" % (nPageIndex+1, oTrain.EstimatedPages, sPageFileName) )     
    #------------------------------------------------------------------------------------
    def Initialize(self):
        self.ClassCount=None
        self.__loadClassesFromDisk()
        self.Train.LoadSampleIndexFromDisk()
        self.Train.LoadFromDisk()
        
        self.Validation.LoadSampleIndexFromDisk()
        self.Validation.LoadFromDisk()
    #------------------------------------------------------------------------------------
                
#==================================================================================================


    