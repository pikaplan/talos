# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        FILE SYSTEM OPERATIONS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import os
import shutil
import tarfile
import pickle
import zipfile
from shutil import copyfile, move
import TALOS.Constants as tcc
 



def OS_PATH_SEPARATOR():
    return os.pathsep




#==================================================================================================
class BaseFolders:
    __verboseLevel = 1
    
    
    RUN = "NNRun"
    NEURAL_NETS = "NNStore"
    RUN_QUEUE = "NNQ"
    MACHINE_LEARNING_DATA = "MLData"
    
    EXPERIMENTS_RUN     = "C:\\NNRun\\"
    EXPERIMENTS_STORE   = "H:\\NNStore\\"
    EXPERIMENTS_SYSTEM  = "H:\\NNQ\\"
    DATASETS            = "H:\\MLData\\"
    DATASETS_SOURCE     = "H:\\MLData\\"
        
    # Maximum usage in GBytes for experiment saved model
    EXPERIMENT_RUN_MAX_DISK_USAGE = 20
    #------------------------------------------------------------------------------------
    @classmethod
    def Set(cls, p_sRunRoot, p_sDataSetsRoot, p_sStoreRoot=None, p_sDataSetsSource=None, p_nRunMaxDiskUsage=None):
        cls.EXPERIMENTS_RUN = os.path.join(os.path.join(p_sRunRoot, cls.RUN), "")
        cls.EXPERIMENTS_STORE = cls.EXPERIMENTS_RUN
        if p_sStoreRoot is not None:
            cls.EXPERIMENTS_STORE = os.path.join(os.path.join(p_sStoreRoot, cls.NEURAL_NETS), "")
            cls.EXPERIMENTS_SYSTEM = os.path.join(p_sStoreRoot, cls.RUN_QUEUE)
            
        cls.DATASETS = os.path.join(os.path.join(p_sDataSetsRoot, cls.MACHINE_LEARNING_DATA), "")
        cls.DATASETS_SOURCE = cls.DATASETS
        if p_sDataSetsSource is not None:  
            cls.DATASETS_SOURCE = os.path.join(os.path.join(p_sDataSetsSource, cls.MACHINE_LEARNING_DATA), "")
            
        if p_nRunMaxDiskUsage is not None:
            cls.EXPERIMENT_RUN_MAX_USAGE = p_nRunMaxDiskUsage
                    
        if cls.__verboseLevel >=2 :
            print("."*20 + "File System" + "."*19)
            print("Experiment run root:%s" %  cls.EXPERIMENTS_RUN)    
            print("Model store root   :%s" % cls.EXPERIMENTS_STORE)
            print("Dataset root       :%s" % cls.DATASETS)
            print("Dataset images root:%s" % cls.DATASETS_SOURCE)
            print("Maximum run disk usage:%d GBytes" % cls.EXPERIMENT_RUN_MAX_DISK_USAGE)
            print("."*50)      
        elif cls.__verboseLevel >=1 :
            print("[TALOS] FS: EXPR:%s | DS:%s | NNQ:%s" %  (cls.EXPERIMENTS_RUN,cls.DATASETS, cls.EXPERIMENTS_SYSTEM))  
                
    #------------------------------------------------------------------------------------    
     
#==================================================================================================    







    
    
    
    
    
    
#==================================================================================================        
class DummyLogger(object):
    #------------------------------------------------------------------------------------
    def Print(self, f_arg, *args, end=None):
        print(f_arg, *args, end=end)
    #------------------------------------------------------------------------------------
#==================================================================================================
    
    


#==================================================================================================
class DataSetFolder(object):
    __verboseLevel=1
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_sBaseFolder=None, p_sSourceFolder=None):
        #........ |  Instance Attributes | ..............................................
        self.BaseFolder=os.path.join(p_sBaseFolder, "")
        Storage.EnsurePathExists(self.BaseFolder)
        
        if p_sSourceFolder is None:
            self.SourceFolder=os.path.join(self.BaseFolder, "source")
        else:
            self.SourceFolder=p_sSourceFolder
        self.SourceFolderLinked=None 
        #assert Storage.IsExistingPath(self.SourceFolder), "Samples source folder %s not found" % self.SourceFolder
        
        self.DictFolder=os.path.join(self.BaseFolder, "dict")
        self.CacheFolder=os.path.join(self.BaseFolder, "data")
                
        self.TrainSourceFolder=os.path.join(self.SourceFolder, "train")
        self.ValSourceFolder=os.path.join(self.SourceFolder, "val")
        self.TestSourceFolder=os.path.join(self.SourceFolder, "test")
        
        
        self.TrainSourceTarFolder=None
        self.ValSourceTarFolder=None
        self.TestSourceTarFolder=None
        self.ValSourceAssortedFolder=None
                
        
        self.TrainCacheFolder=os.path.join(self.CacheFolder, "train")
        self.ValCacheFolder=os.path.join(self.CacheFolder, "val")
        self.TestCacheFolder=os.path.join(self.CacheFolder, "test")

        self.ClassesFile=os.path.join(self.DictFolder, "classes.dict")
        #self.TrainClassFoldersFile=os.path.join(self.DictFolder, "train.class.folders.dict")
        

        self.ClassNames=[]
        self.ClassFolders=[]
        self.ClassCodesDict=dict()
        
        self.__pathsToEnsure=None    
        self.__validationSampleClasses=None
        self.__validationSampleFileNames=None    
        #................................................................................
        self.__followLinkToSourceFolder()
        
        # Ensures the existence of metadata folders
        Storage.EnsurePathExists(self.DictFolder)

        #Storage.EnsurePathExists(self.TrainSourceFolder)
        #Storage.EnsurePathExists(self.ValSourceFolder)
        #Storage.EnsurePathExists(self.TestSourceFolder)
        
        # Ensures the existence of pickle cache folders
        Storage.EnsurePathExists(self.CacheFolder)
        Storage.EnsurePathExists(self.TrainCacheFolder)
        Storage.EnsurePathExists(self.ValCacheFolder)
        Storage.EnsurePathExists(self.TestCacheFolder)      
        
        # Default files names
        self.ClassNamesFile=os.path.join(self.SourceFolder, "classinfo.txt")
    #------------------------------------------------------------------------------------
    def __followLinkToSourceFolder(self):
        self.__pathsToEnsure = []
        
        self.SourceFolderLinked=None
        sLinkFileName = os.path.join(self.SourceFolder, "link_path.ini")
        if os.path.isfile(sLinkFileName):
            with open(sLinkFileName, "r") as oFile:
                sLines = oFile.readlines()
                if len(sLines) >= 0:
                    self.SourceFolderLinked=sLines[0].strip()

            
        if self.SourceFolderLinked is not None:
            self.TrainSourceTarFolder=os.path.join(self.SourceFolderLinked, "train.tar")
            self.ValSourceTarFolder=os.path.join(self.SourceFolderLinked, "val.tar")
            self.TestSourceTarFolder=os.path.join(self.SourceFolderLinked, "test.tar")
            self.ValSourceAssortedFolder=os.path.join(self.SourceFolderLinked, "val.all")
            
            self.ValLabelsFile=os.path.join(os.path.join(self.SourceFolderLinked, "labels"), "val.txt")
                    
            self.TrainSourceFolder=os.path.join(self.SourceFolderLinked, "train")
            self.ValSourceFolder=os.path.join(self.SourceFolderLinked, "val")
            self.TestSourceFolder=os.path.join(self.SourceFolderLinked, "test")

            self.__pathsToEnsure.append(self.TrainSourceFolder)
            self.__pathsToEnsure.append(self.ValSourceFolder)
            self.__pathsToEnsure.append(self.TestSourceFolder)
            
        for sFolder in self.__pathsToEnsure:
            Storage.EnsurePathExists(sFolder)  
    #------------------------------------------------------------------------------------
    def GetSampleIndexFile(self, p_nType):
        sFileName=None
        if p_nType == tcc.DS_TRAINING:
            sFileName = "train.samples.dict"
        elif p_nType == tcc.DS_VALIDATION:
            sFileName = "val.samples.dict"
        elif p_nType == tcc.DS_TESTING:
            sFileName = "test.samples.dict"
        else:
            raise Exception("Invalid sample set type %d" % p_nType) 
                
        return os.path.join(self.DictFolder, sFileName)
    #------------------------------------------------------------------------------------
    def GetSampleSetFile(self, p_nType):
        sFileName=None
        if p_nType == tcc.DS_TRAINING:
            sFileName = "train.dict"
        elif p_nType == tcc.DS_VALIDATION:
            sFileName = "val.dict"
        elif p_nType == tcc.DS_TESTING:
            sFileName = "test.dict"
        else:
            raise Exception("Invalid sample set type %d" % p_nType)
            
        return os.path.join(self.DictFolder, sFileName) 
    #------------------------------------------------------------------------------------
    def GetPageFolder(self, p_nType):
        sFolder = None
        if p_nType == tcc.DS_TRAINING:
            sFolder = self.TrainCacheFolder
        elif p_nType == tcc.DS_VALIDATION:
            sFolder = self.ValCacheFolder
        elif p_nType == tcc.DS_TESTING:
            sFolder = self.TestCacheFolder
        else:
            raise Exception("Invalid sample set type %d" % p_nType)
        
        return sFolder        
    #------------------------------------------------------------------------------------
    #TO_DEPRECATE
    def InitializeForDecompression(self, p_sSourceTarFolder, p_sDestImagesFolder):
        
        
        self.SourceTarFolder=p_sSourceTarFolder
        self.TrainSourceTarFolder=os.path.join(self.SourceTarFolder, "train")
        self.ValSourceTarFolder=os.path.join(self.SourceTarFolder, "val")
        self.TestSourceTarFolder=os.path.join(self.SourceTarFolder, "test")

        self.ClassNamesFile=os.path.join(self.SourceTarFolder, "words.txt")
        #TEMP
        self.ImageNetClassesFile=os.path.join(os.path.join(self.SourceTarFolder, "labels"), "synset_words.txt")
        
        
        if p_sDestImagesFolder is not None:
            # TEMP: Support validation
            self.TrainSourceFolder = os.path.join(p_sDestImagesFolder, "train")
    #------------------------------------------------------------------------------------
    def __loadValidationSampleClasses(self):
        nCollection=[]
        with open(self.ValLabelsFile, "r") as oFile:
            for sLine in oFile:
                sParts = sLine.split(" ")
                sFileName = Storage.JoinPath(self.ValSourceAssortedFolder, sParts[0].strip())
                nClassIndex = int(sParts[1].strip())
                nCollection.append([nClassIndex, sFileName])
                print("[%d] %s" % (nClassIndex, sFileName))
                         
        return nCollection
    #------------------------------------------------------------------------------------
    def CollectValidationSamples(self, p_nClassIndex, p_sClassCode, p_nType=tcc.DS_VALIDATION):
        if self.__validationSampleClasses is None:
            self.__validationSampleClasses = self.__loadValidationSampleClasses()
        
        sTargetFolder = self.GetClassFolder(p_sClassCode, p_nType)
        
        Storage.EnsurePathExists(sTargetFolder)
        print("Copying to %s ..." % sTargetFolder)
            
        for oValRec in self.__validationSampleClasses:
            nClassIndex = oValRec[0]
            sFileName = oValRec[1]
            if nClassIndex == p_nClassIndex:
                Storage.CopyFileToFolder(sFileName, sTargetFolder)
        return sTargetFolder
    #------------------------------------------------------------------------------------
    def DecompressTrainSamples(self):
        sSortedFiles = Storage.GetFilesSorted(self.TrainSourceTarFolder)
        
        nCount=0
        for sFileName in sSortedFiles:
            _, sFileNameOnly, sFileExt = Storage.SplitFileName(sFileName)
            bContinue = (sFileExt == "tar")
                            
            #TODO: Condition that the filename
            if bContinue: 
                bContinue = nCount < 10
                #bContiue = sFileNameOnly in sSelectedCodes
               
            if bContinue:
                sSourceFileName = os.path.join(self.TrainSourceTarFolder, sFileName) 
                sTargetFolder = os.path.join(self.TrainSourceFolder, sFileNameOnly)
                
                if not Storage.IsExistingPath(sTargetFolder):
                    print("%i: %s" % (nCount, sTargetFolder))
                    tar = tarfile.open(sSourceFileName, "r:")
                    tar.extractall(sTargetFolder) 
                    tar.close()
                       
                nCount += 1
    #------------------------------------------------------------------------------------
    def DecompressSamples(self, p_sClassCode, p_nType=tcc.DS_TRAINING):
        sSourceFileName = os.path.join(self.TrainSourceTarFolder, p_sClassCode + ".tar")
        sTargetFolder = self.GetClassFolder(p_sClassCode, p_nType)
        
        if (not Storage.IsExistingPath(sTargetFolder)):
            if type(self).__verboseLevel >=1:
                print("    Extracting %s to %s ..." % (sSourceFileName, sTargetFolder))
            tar = tarfile.open(sSourceFileName, "r:")
            tar.extractall(sTargetFolder) 
            tar.close()   
            
        return sTargetFolder     
    #------------------------------------------------------------------------------------
    def GetClassFolder(self, p_sClassCode, p_nType):
        sResult = None
        if p_nType == tcc.DS_TRAINING:
            sResult = os.path.join(os.path.join(self.TrainSourceFolder, p_sClassCode), "")
        elif p_nType == tcc.DS_VALIDATION:
            sResult = os.path.join(os.path.join(self.ValSourceFolder, p_sClassCode), "") 
        elif p_nType == tcc.DS_TESTING:
            sResult = os.path.join(os.path.join(self.TestSourceFolder, p_sClassCode), "") 
        else:
            raise Exception("Invalid machine learning data set type %d" % p_nType)
            
        return sResult
#==================================================================================================






#==================================================================================================
class ModelsFolder(object):
    __verboseLevel = 2
    #------------------------------------------------------------------------------------
    def __init__(self, p_sFolder, p_nModelStoreLimitInGBs=None):
        #................... |  Instance Attributes | ...................................
        self.Folder = p_sFolder
        if p_nModelStoreLimitInGBs is None:
            self.ModelStoreLimitInMBs = BaseFolders.EXPERIMENT_RUN_MAX_DISK_USAGE * 1024.0
        else:
            self.ModelStoreLimitInMBs = p_nModelStoreLimitInGBs * 1024.0
        self.InitialStateFiles=None
        self.ModelSizeInMBs = None #In MBs
        self.MaxModels = None
        self.Models=[]
        self.ModelsOld=[]
        self.Log=DummyLogger()
        #................................................................................
    #------------------------------------------------------------------------------------
    def __calcInitialStateFilesSize(self):
        if self.MaxModels is None:
            self.InitialStateFiles = Storage.GetFilesSorted(self.Folder)
            
            bContinue = True
            if (self.InitialStateFiles is None) or (self.InitialStateFiles == []):
                bContinue = False
            
            if bContinue:
                nModelSizeInByte = 0
                
                for nFile in self.InitialStateFiles:
                    nSize = os.path.getsize(os.path.join(self.Folder, nFile))
                    nModelSizeInByte += nSize
                    
                self.ModelSizeInMBs = nModelSizeInByte / (1024.0**2)
                self.MaxModels = int( self.ModelStoreLimitInMBs / self.ModelSizeInMBs )
            
                if type(self).__verboseLevel >=1 :
                    self.Log.Print("  |__ Model Folder:%s  Size of Model:%.1fMBs  Maximum:%d" % (self.Folder, self.ModelSizeInMBs, self.MaxModels))
    #------------------------------------------------------------------------------------
    def Refresh(self):
        sSubFolders = Storage.GetDirectoriesSorted(self.Folder)
        
        self.Models=[]
        for sFolder in sSubFolders:
            sSubFolder = os.path.join(self.Folder, sFolder)
            nEpochNumber = int(sFolder)
            self.Models.append([nEpochNumber - 1, sSubFolder])
        self.ModelsOld = list(self.Models)
    #------------------------------------------------------------------------------------
    def EnsureMaxModels(self, p_nEpochScores, p_nBestIsHighest=True):
        if self.MaxModels is None:
            return
        
        nDeletedCount = 0
        self.Refresh()
        while len(self.Models) > self.MaxModels:
            nDeletedEpochIndex = self.RemoveWorstModel(p_nEpochScores, p_nBestIsHighest)
            if nDeletedEpochIndex is not None:
                nDeletedCount += 1
                
        if type(self).__verboseLevel >= 2:
            for nRec in self.Models:
                self.Log.Print("%d " % nRec[0], end="")
            print("")
        
        print(nDeletedCount)
    #------------------------------------------------------------------------------------
    def RemoveWorstModel(self, p_nEpochScores, p_nBestIsHighest=True):
        sModelFolderToDelete = None
        nResult = None
        
        nWorstValue = None
        nWorstIndex = None
        nWorstEpochIndex = None
        for nIndex, nModelRec in enumerate(self.Models):
            nModelEpochIndex = nModelRec[0]
            nScore = p_nEpochScores[nModelEpochIndex]
            
            # Do not remove the last model
            if nIndex < len(self.Models) - 1:
                if nWorstValue is None:
                    nWorstValue = nScore
                    nWorstIndex = nIndex
                    nWorstEpochIndex = nModelEpochIndex
                elif p_nBestIsHighest and nScore < nWorstValue:
                    nWorstValue = nScore
                    nWorstIndex = nIndex
                    nWorstEpochIndex = nModelEpochIndex
                elif (not p_nBestIsHighest) and nScore > nWorstValue:
                    nWorstValue = nScore
                    nWorstIndex = nIndex
                    nWorstEpochIndex = nModelEpochIndex
        
            
        if nWorstIndex is not None:
            sModelFolderToDelete = self.Models[nWorstIndex][1]
            self.Log.Print("  [>] Removing saved model for epoch %.3d: %s" % (nWorstEpochIndex + 1, sModelFolderToDelete))
            del self.Models[nWorstIndex]
            nResult = nWorstEpochIndex      
        #if sModelFolderToDelete is not None:
        #    Storage.RemoveFolder(sModelFolderToDelete)
        return nResult
    #------------------------------------------------------------------------------------
    def Initialize(self, p_oLogFile=None):
        if p_oLogFile is not None:
            self.Log = p_oLogFile
        self.__calcInitialStateFilesSize()
        return self.InitialStateFiles 
    #------------------------------------------------------------------------------------
#==================================================================================================








    
   
#==================================================================================================
class Storage(object):
    #------------------------------------------------------------------------------------
    @classmethod
    def JoinPath(cls, p_sPrefix, p_sSuffix):
        return os.path.join(p_sPrefix, p_sSuffix)
    #------------------------------------------------------------------------------------
    @classmethod
    def JoinPaths(cls, p_sPaths):
        sResult = ""
        for sPath in p_sPaths:
            sResult =  os.path.join(sResult, sPath)
        
        return sResult
    #------------------------------------------------------------------------------------
    @classmethod
    def IsExistingPath(cls, p_sPath):
        return os.path.exists(p_sPath)
    #------------------------------------------------------------------------------------
    @classmethod
    def IsExistingFile(cls, p_sPath, p_sFileOnPath=None):
        if p_sFileOnPath is not None:
            sFileName = os.path.join(p_sPath, p_sFileOnPath)
        else:
            sFileName = p_sPath
            
        return os.path.isfile(sFileName)  
    #------------------------------------------------------------------------------------
    def CreateFlagFile(self, p_sPath, p_sFileOnPath=None, p_sContent=None):
        if p_sFileOnPath is not None:
            sFileName = os.path.join(p_sPath, p_sFileOnPath)
        else:
            sFileName = p_sPath        
                     
        with open(sFileName, "w") as oFile:
            if p_sContent is not None:
                print(p_sContent, file=oFile)
            else:
                print(".", file=oFile)            
            oFile.close()
    #------------------------------------------------------------------------------------
    @classmethod
    def SplitFileName(cls, p_sPath):
        path, filename_w_ext = os.path.split(p_sPath)
        filename, file_extension = os.path.splitext(filename_w_ext)        
        return path, filename, file_extension
    #------------------------------------------------------------------------------------
    @classmethod
    def JoinFileName(cls, p_sPath, p_sFile, p_sExt):
        sFileName = os.path.join(p_sPath, p_sFile + p_sExt)
        return sFileName  
    #------------------------------------------------------------------------------------
    @classmethod
    def SplitLastFolder(cls, p_sPath):
        return os.path.basename(os.path.normpath(p_sPath))
    #------------------------------------------------------------------------------------
    @classmethod
    def EnsurePathExists(cls, p_sPath):
        """
        Ensures that p_sFolderName path exists
        
        Returns: True: if path already exists, False if path was not found and created
        """
        #TODO: Recursive for many dirs if python don't does this automatically
        Result=True
        if not os.path.exists(p_sPath):
            os.makedirs(p_sPath)
            Result=False
            
        return Result     
    #------------------------------------------------------------------------------------
    @classmethod    
    def EnsureValidFileName(cls, p_sFileName):
        translation_table = str.maketrans('?:*/|"<>\\', "!.+-I'---")
        Result = p_sFileName.translate(translation_table)
        #print(p_sFileName, Result)
        return Result
    #----------------------------------------------------------------------------------
    @classmethod    
    def IsFolderEmpty(cls, p_sFolderName):
        return os.listdir(p_sFolderName) == [] 
    #----------------------------------------------------------------------------------
    @classmethod
    def ExistFilesOnPath(cls, p_sFolderName, p_sFileNameOnlyList):
        bResult=True
        if not os.path.exists(p_sFolderName):
            bResult=False
        
        if bResult:
            for sFileName in p_sFileNameOnlyList:
                if not os.path.isfile(p_sFolderName + sFileName):
                    bResult=False
                    break
            
        return bResult
    #----------------------------------------------------------------------------------
    @classmethod
    def GetDirectoriesSorted(cls, p_sFolderName):
        return sorted(next(os.walk(p_sFolderName))[1])
    #----------------------------------------------------------------------------------    
    @classmethod
    def GetFilesSorted(cls, p_sFolderName):
        return sorted(next(os.walk(p_sFolderName))[2])
    #------------------------------------------------------------------------------------
    @classmethod
    def SplitPath(cls, p_sPath):
        sParentFolder = os.path.dirname(os.path.normpath(p_sPath))
        sSubFolder    = os.path.basename(os.path.normpath(p_sPath))
        return sParentFolder, sSubFolder        
    #------------------------------------------------------------------------------------
    @classmethod
    def CompressFolder(cls, p_sFolderName, p_sDestZipName, p_bIsDisplayingFileProgress=False):
        if not cls.IsExistingPath(p_sFolderName):
            return False, None
        
        sParentFolder, sSubFolder = cls.SplitPath(p_sFolderName)
        
        if p_sDestZipName is not None:
            sDestFolder, sFileName, sExt = Storage.SplitFileName(p_sDestZipName)
            sZipFileName = sFileName + sExt
        else:
            sDestFolder = None
            sZipFileName = sSubFolder + ".zip"
            
        sZipFullPath = os.path.join(sParentFolder, sZipFileName)
        
        sFiles = sorted(next(os.walk(p_sFolderName))[2])
        
        print("  {.} Compressing folder %s into [%s] ..." % (p_sFolderName, sZipFullPath))
        bCanCreateZip = len(sFiles) > 0
        if bCanCreateZip:
            with zipfile.ZipFile(sZipFullPath, "w", zipfile.ZIP_DEFLATED) as oZip:
                for sFile in sFiles:
                    sSourceFileName  = os.path.join(p_sFolderName, sFile)
                    sArchiveFileName = os.path.join(sSubFolder, sFile)
                    if p_bIsDisplayingFileProgress:
                        print("   |__ %s -> %s" % (sSourceFileName, sArchiveFileName))
                    oZip.write(sSourceFileName, sArchiveFileName)
                oZip.close()   
                
            if (sDestFolder is not None) and (sDestFolder.strip() != ""):
                cls.EnsurePathExists(sDestFolder)
                print("  {.} Moving [%s] into folder %s ..." % (sZipFullPath, sDestFolder ))
                cls.MoveFileToFolder(sZipFullPath, sDestFolder)
                
        return bCanCreateZip, sZipFullPath
    #------------------------------------------------------------------------------------
    @classmethod
    def DecompressFile(cls, p_sFileName, p_sDestFolder=None):
        bCanDecompress = os.path.isfile(p_sFileName)
        if bCanDecompress:
            if p_sDestFolder is None:
                sDestFolder, _, _ = Storage.SplitFileName(p_sFileName)
            else:
                sDestFolder = p_sDestFolder
                 
            print("  {.} Decompressing file [%s] into %s ..." % (p_sFileName, sDestFolder))
            with zipfile.ZipFile(p_sFileName, "r") as oZip:
                oZip.extractall(sDestFolder)
                oZip.close()    
        return bCanDecompress
    #------------------------------------------------------------------------------------
    @classmethod
    def DeleteFolderFiles(cls, p_sFolder, p_bIsPrintingMessage=True):
        if p_bIsPrintingMessage:
            print("  {.} Deleting * in folder %s" % p_sFolder)
        sModelFiles = Storage.GetFilesSorted(p_sFolder)
        for sFile in sModelFiles:
            sFileNameFull = os.path.join(p_sFolder, sFile)
            if os.path.isfile(sFileNameFull):
                os.remove(sFileNameFull)
    #------------------------------------------------------------------------------------
    @classmethod
    def DeleteEmptyFolder(cls, p_sFolderName):
        if os.path.exists(p_sFolderName):
            bIsEmpty = os.listdir(p_sFolderName) == []
            if bIsEmpty:
                shutil.rmtree(p_sFolderName)
    #------------------------------------------------------------------------------------
    @classmethod
    def DeleteFile(cls, p_sFileName):
        if os.path.isfile(p_sFileName):
            os.remove(p_sFileName)
    #------------------------------------------------------------------------------------
    @classmethod
    def RemoveFolder(cls, p_sFolder):
        bMustRemove = os.path.exists(p_sFolder)
        if bMustRemove:
            print("  {.} Removing folder %s" % p_sFolder)
            Storage.DeleteFolderFiles(p_sFolder, p_bIsPrintingMessage=False)
            os.removedirs(p_sFolder) 
        else:
            print("   -  Folder %s not found" % p_sFolder)
    # ----------------------------------------------------------------------------------
    @classmethod
    def CopyFile(cls, p_sSourceFileName, p_sDestFileName, p_bIsOverwriting=False):
        if p_bIsOverwriting:
            if os.path.exists(p_sDestFileName):
                os.remove(p_sDestFileName)
            copyfile(p_sSourceFileName, p_sDestFileName)
        else:
            if not os.path.exists(p_sDestFileName):
                copyfile(p_sSourceFileName, p_sDestFileName)
    # ----------------------------------------------------------------------------------
    @classmethod
    def CopyFileToFolder(cls, p_sSourceFileName, p_sDestFolder, p_bIsOverwriting=False):
        _, sFileNameWithExtension = os.path.split(p_sSourceFileName)
        Storage.CopyFile(p_sSourceFileName, os.path.join(p_sDestFolder, sFileNameWithExtension), p_bIsOverwriting=p_bIsOverwriting)
    #----------------------------------------------------------------------------------
    @classmethod
    def MoveFile(cls, p_sSourceFileName, p_sDestFileName):
        if os.path.isfile(p_sSourceFileName):
            move(p_sSourceFileName, p_sDestFileName)
    #----------------------------------------------------------------------------------    
    @classmethod
    def MoveFileToFolder(cls, p_sSourceFileName, p_sDestFolder, p_sDestFileName=None):
        if os.path.isfile(p_sSourceFileName):
            if p_sDestFileName is not None:
                sFileNameWithExtension = p_sDestFileName
            else:
                _, sFileNameWithExtension = os.path.split(p_sSourceFileName)

            move(p_sSourceFileName, os.path.join(p_sDestFolder, sFileNameWithExtension))
    #----------------------------------------------------------------------------------    
    @classmethod
    def GetFileNameNextVersion(cls, p_sFileName):
        """
        Filename versioning support: Gets the next version of the given filename
        """
        sNextVersionFileName=None
        for nIndex in range(0,1000):
            sNextVersionFileName=p_sFileName[:-4] + "_r%03i" % nIndex + p_sFileName[-4:]
            if not os.path.isfile(sNextVersionFileName):
                break
        return sNextVersionFileName
    #----------------------------------------------------------------------------------
    @classmethod
    def GetFileNameLastVersion(cls, p_sFileName):
        """
        Filename versioning support: Gets the last version of the given filename
        """ 
           
        sLastVersionFileName=None
        if os.path.isfile(p_sFileName):
            sLastVersionFileName=p_sFileName
        else:
            for nIndex in range(0,1000):
                sFileName=p_sFileName[:-4] + "_r%03i" % nIndex + p_sFileName[-4:]
                if not os.path.isfile(sFileName):
                    break
                sLastVersionFileName=sFileName
            
        return sLastVersionFileName
    #----------------------------------------------------------------------------------
    @classmethod
    def DeserializeObjectFromFile(cls, p_sFileName, p_bIsVersioned=False, p_bIsPython2Format=False, p_bIsVerbose=True):
        """
        Deserializes the data from a pickle file if it exists.
        Parameters
            p_sFileName        : Full path to the  python object file 
        Returns
            The object with its data or None when the file is not found.
        """
        oData=None
        
        if p_bIsVersioned:
            p_sFileName=cls.GetFileNameLastVersion(p_sFileName)
    
        if os.path.isfile(p_sFileName):
            if p_bIsVerbose:
                print("      {.} Loading data from %s" % p_sFileName)
            with open(p_sFileName, "rb") as oFile:
                if p_bIsPython2Format:
                    oUnpickler = pickle._Unpickler(oFile)
                    oUnpickler.encoding = 'latin1'
                    oData =oUnpickler.load()
                else:
                    oData = pickle.load(oFile)
                oFile.close()
                
        
        return oData
    #----------------------------------------------------------------------------------
    @classmethod    
    def SerializeObjectToFile(cls, p_sFileName, p_oData, p_bIsOverwritting=False, p_bIsVersioned=False, p_nExtraLabel=None):    
        """
        Serializes the data to a pickle file if it does not exists.
        Parameters
            p_sFileName        : Full path to the  python object file 
        Returns
            True if a new file was created
        """
        bResult=False
        if p_bIsVersioned:
            bMustContinue = True
            p_sFileName=cls.GetFileNameNextVersion(p_sFileName)
        else:
            if p_bIsOverwritting:
                bMustContinue = True
            else:
                bMustContinue = not os.path.isfile(p_sFileName)
        
        if bMustContinue:
            if p_nExtraLabel is not None:
                print("  {%s} Saving data to %s" % (p_nExtraLabel, p_sFileName) )                    
            else:
                print("  {.} Saving data to %s" % p_sFileName)
            with open(p_sFileName, "wb") as oFile:
                pickle.dump(p_oData, oFile, pickle.HIGHEST_PROTOCOL)
                oFile.close()
            bResult=True
        else:
            if p_nExtraLabel is not None:
                print("  {%s} Not overwritting %s" % (p_nExtraLabel, p_sFileName) )                    
            else:
                print("  {.} Not overwritting %s" % p_sFileName)
            
            
                            
        return bResult
    #----------------------------------------------------------------------------------
    @classmethod 
    def GetFolderSize(cls, p_sFolder):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(p_sFolder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
                
        return total_size
    #----------------------------------------------------------------------------------
    @classmethod
    def ReadTextFile(cls, p_sFileName):
        sResult = []
        if os.path.isfile(p_sFileName):
            with open(p_sFileName) as oInFile:
                oData = oInFile.readlines()
            
            for sLine in enumerate(oData):
                sResult.append(sLine[1].rstrip())
                  
        return sResult
    #----------------------------------------------------------------------------------
    @classmethod
    def CurrentFolder(cls):
        return os.getcwd()  
    #----------------------------------------------------------------------------------
    
#==================================================================================================