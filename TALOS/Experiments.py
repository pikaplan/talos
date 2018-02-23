# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.1.0-ALPHA
#        MACHINE LEARNING EXPERIMENTS 
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import tensorflow as tf
import os
import sys
import json
import random
import numpy as np
import TALOS.Constants as tcc
from TALOS.Core import Logger, MinuteUID, SysParams, Exec, ERLString
from TALOS.Utils import OperatingSystemSignature
from TALOS.FileSystem import Storage, BaseFolders
from TALOS.Metrics import ClassificationMetrics, ClassificationBest
from TALOS.HyperParams import NNLearnConfig








    

    
    


    
#------------------------------------------------------------------------------------
def GetModelAndFoldCommandArguments(p_sDefaultModelName=None, p_nDefaultFoldNumber=10, p_nDefaultIsEvaluating=False):
    sModelName=p_sDefaultModelName
    nFoldNumber=p_nDefaultFoldNumber
    bIsEvaluating=p_nDefaultIsEvaluating
    
    if len(sys.argv) > 1:
        sModelName=sys.argv[1]
        if len(sys.argv) > 2:
            nFoldNumber=int(sys.argv[2])
        if len(sys.argv) > 3:
            bIsEvaluating=(int(sys.argv[3]) == 1)
        
    return sModelName, nFoldNumber, bIsEvaluating
#------------------------------------------------------------------------------------








#==================================================================================================
class MonitoringSettings(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sFileName):
        #........ |  Instance Attributes | ..............................................
        self.FileName = p_sFileName        
        self.ModelsToCompare=None
        self.ModelTitles=None
        self.ModelBaseFolders=None
        #................................................................................
        self.Load()
    #------------------------------------------------------------------------------------        
    def Load(self):
        if self.FileName is not None:
            with open(self.FileName) as oDefFile:    
                dJSON = json.load(oDefFile)
                
        self.ModelsToCompare=[]
        self.ModelTitles=[]
        self.ModelBaseFolders=[]
                
        oModels=dJSON["CompareModels"]
        for oModel in oModels:
            print("Adding to comparison:", oModel[0], ":", "'%s'" % oModel[1])
            self.ModelsToCompare.append(oModel[0])
            self.ModelTitles.append(oModel[1])
            self.ModelBaseFolders.append(oModel[2])
    #------------------------------------------------------------------------------------        
#==================================================================================================



    



#==================================================================================================
class StatsColumnType(object):
    VALUE               = 0
    DELTA_VALUE         = 1
    ARCTAN_DELTA_VALUE  = 2
    COUNT = 3
    
    
    @classmethod
    def ToString(cls, p_sBaseDescr, p_nType):
        p_sBaseDescr = p_sBaseDescr.replace(" ", "")
        if p_nType == StatsColumnType.VALUE:
            return p_sBaseDescr
        elif p_nType == StatsColumnType.DELTA_VALUE:
            return "Δ%s" % p_sBaseDescr
        elif p_nType == StatsColumnType.ARCTAN_DELTA_VALUE:
            return "ArcTan(Δ%s)" % p_sBaseDescr
        return  
    
#==================================================================================================
    
    
    
    
    
    
#TODO: 0.7 Setup file for testing different DNAs    
    
#==================================================================================================
class ExperimentSubFolder(object):
    NO_SUBFOLDERS="{None}"
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nFoldNumber, p_bIsRun=False):
        #........ |  Instance Attributes | ..............................................
        self.ParentExperiment = p_oParent
        self.FoldNumber = p_nFoldNumber
        self.IsRun = p_bIsRun
        if self.IsRun:
            self.Folder      = os.path.join(self.ParentExperiment.RunBaseFolder, "fold%.2d" % self.FoldNumber )
        else:
            self.Folder      = os.path.join(self.ParentExperiment.BaseFolder   , "fold%.2d" % self.FoldNumber )
        Storage.EnsurePathExists(self.Folder)   
            
        sFolders = Storage.GetDirectoriesSorted(self.Folder)
        if len(sFolders) > 0:   
            self.LastUID = sFolders[-1]
        else:
            self.LastUID = ExperimentSubFolder.NO_SUBFOLDERS
        
        self.__pathsToEnsure=None        
        #self.__initExperimentFolder(self.LastUID)
        #................................................................................
    #------------------------------------------------------------------------------------
    def __defineExperimentFolders(self, p_sUID):
        self.__pathsToEnsure=[]
        self.ExperimentFolder           = os.path.join(self.Folder, p_sUID)
        self.ExperimentLogFolder        = os.path.join(self.ExperimentFolder, "log")
        self.ExperimentStatsFolder      = os.path.join(self.ExperimentFolder, "stats")
        self.ExperimentResultsFolder    = os.path.join(self.ExperimentFolder, "results")
        self.ExperimentModelFolder      = os.path.join(self.ExperimentFolder, "models")
        self.ExperimentGraphsFolder     = os.path.join(self.ExperimentFolder, "graphs")
        self.ExperimentPlotFolder       = os.path.join(self.ExperimentGraphsFolder, "plots")
        self.ExperimentVisualsFolder    = os.path.join(self.ExperimentFolder, "visuals")
        self.ExperimentConfigFolder     = os.path.join(self.ExperimentFolder, "config")
        self.ExperimentBestsFolder      = os.path.join(self.ExperimentFolder, "best")
        self.ExperimentLogSamplesFolder = os.path.join(self.ExperimentLogFolder, "samples")
        assert self.ParentExperiment.ModelName is not None, "Experiment model architecture is not defined"
        self.ArchitectureCommonFolder    = os.path.join(self.ParentExperiment.RunBaseFolder, "common")
        
        self.__pathsToEnsure.append(self.ExperimentFolder)
        self.__pathsToEnsure.append(self.ExperimentLogFolder)
        self.__pathsToEnsure.append(self.ExperimentStatsFolder)
        self.__pathsToEnsure.append(self.ExperimentResultsFolder)
        self.__pathsToEnsure.append(self.ExperimentModelFolder)
        self.__pathsToEnsure.append(self.ExperimentGraphsFolder)
        self.__pathsToEnsure.append(self.ExperimentPlotFolder)
        self.__pathsToEnsure.append(self.ExperimentVisualsFolder)
        self.__pathsToEnsure.append(self.ExperimentConfigFolder)
        self.__pathsToEnsure.append(self.ExperimentBestsFolder)
        self.__pathsToEnsure.append(self.ExperimentLogSamplesFolder)
        self.__pathsToEnsure.append(self.ArchitectureCommonFolder)
        
        
        # // Saved model file system names \\
        self.CommonInitialModelFileName   = os.path.join(self.Folder                    , "init.nn")
        self.InitialModelFileName         = os.path.join(self.ExperimentModelFolder     , "init_%s.nn" % p_sUID)
        self.InitialModelZipFileName      = os.path.join(self.ArchitectureCommonFolder  , "initial-model_%s.zip" % p_sUID)
        self.ModelFolderTemplate          = os.path.join(self.ExperimentModelFolder     , "%.3d" )
        self.ModelFileNameTemplate        = os.path.join(self.ModelFolderTemplate       , "model_%s.nn" % p_sUID)
        self.BestModelFileName            = os.path.join(self.ExperimentBestsFolder     , "best_%s.nn" % p_sUID)
        self.InfoFileName                 = os.path.join(self.ExperimentFolder          , "info.dat")        
        # // Evaluation results file system names \\
        self.ModelResultsFileNameTemplate = os.path.join(self.ExperimentResultsFolder   , "%.3d.pkl")
        self.BestModelResultsFileName     = os.path.join(self.ExperimentBestsFolder     , "best_%s.pkl" % p_sUID)
        self.BestModelTextFileName        = os.path.join(self.ExperimentBestsFolder     , "best_%s.txt" % p_sUID)
        #self.BestEvaluationFileName = os.path.join(self.ExperimentBestsFolder       , "best_evaluation_%s.csv" % p_sUID)
        
        self.NetworkArchitectureFileName = os.path.join(self.ExperimentConfigFolder     , "architecture_%s.csv" % p_sUID)
        self.LogFileName                 = os.path.join(self.ExperimentLogFolder        , "log_%s.txt" % p_sUID)
        self.StatsFileName               = os.path.join(self.ExperimentStatsFolder      , "stats_%s.dat" % p_sUID)
        self.StatsTempFileName           = os.path.join(self.ExperimentStatsFolder      , "stats_%s.tmp" % p_sUID)
        self.MarginSamplesTemplate       = os.path.join(self.ExperimentLogSamplesFolder , "sample-margins-%.3d.txt")
        self.MarginHistogramTemplate     = os.path.join(self.ExperimentGraphsFolder     , "histogram-margins-%.3d.png")
        
        self.LearnConfigFileName         = os.path.join(self.ExperimentConfigFolder     , "learn-config-source.cfg")
        self.LearnConfigUsedFileName     = os.path.join(self.ExperimentConfigFolder     , "learn-config-used-%s.cfg" % p_sUID)
    #------------------------------------------------------------------------------------
    def Initialize(self, p_sUID, p_bIsEnsuringPaths=True):
        self.__defineExperimentFolders(p_sUID)
        if p_bIsEnsuringPaths:
            for sFolder in self.__pathsToEnsure:
                Storage.EnsurePathExists(sFolder)    
    #------------------------------------------------------------------------------------
    def ListSavedResults(self):
        if Storage.IsExistingPath(self.ExperimentResultsFolder):
            sModelResultFiles = Storage.GetFilesSorted(self.ExperimentResultsFolder)
            
        oModelResults = []
        for sResultFile in sModelResultFiles:
            _, sFileName, _ = Storage.SplitFileName(sResultFile)
            nEpochNumber = int(sFileName)
            oModelResults.append([nEpochNumber, sResultFile, None])
                        
        return oModelResults
    #------------------------------------------------------------------------------------
    def ListSavedModels(self):
        sModelFolders = []
        if Storage.IsExistingPath(self.ExperimentModelFolder):
            if not Storage.IsFolderEmpty(self.ExperimentModelFolder):
                sModelFolders = Storage.GetDirectoriesSorted(self.ExperimentModelFolder)
        
        
        oModels = []
        for sModel in sModelFolders:
            sFolder = Storage.JoinPath(self.ExperimentModelFolder, sModel)
            sModelFiles = Storage.GetFilesSorted(sFolder)
            nEpochNumber = int(sModel)
            oModels.append([nEpochNumber, sFolder, sModelFiles])
            
            
        return oModels
    #------------------------------------------------------------------------------------
    def ListCompressedModels(self):
        sResult = []
        
        if Storage.IsExistingPath(self.ExperimentModelFolder):
            sModelZipFiles = Storage.GetFilesSorted(self.ExperimentModelFolder)
            for sZipFile in sModelZipFiles:
                sZipFile = Storage.JoinPath(self.ExperimentModelFolder, sZipFile)
                sResult.append(sZipFile)
                 
        return sResult
    #------------------------------------------------------------------------------------
    def DiscardModels(self, p_nEpochNumbers):
        for nEpochNumberToDelete in p_nEpochNumbers:
            self.DeleteSavedModel(nEpochNumberToDelete)
    #------------------------------------------------------------------------------------
    def CompressModels(self, p_nEpochNumbers):
        sUID = self.ParentExperiment.MinuteUID.UID
        for nEpochToCompress in p_nEpochNumbers:
            sModelFolder = self.ModelFolderTemplate % nEpochToCompress
            bContinueToDelete, sArchiveName = Storage.CompressFolder(sModelFolder, "model_%s_epoch_%.3d.zip" % (sUID, nEpochToCompress))
            
            if bContinueToDelete:
                bContinueToDelete = Storage.IsExistingFile(sArchiveName)
            
            if bContinueToDelete:
                self.DeleteSavedModel(nEpochToCompress)
    #------------------------------------------------------------------------------------
    def DeleteSavedModel(self, p_nEpochNumber):
        sModelFolder = self.ModelFolderTemplate % p_nEpochNumber
        Storage.RemoveFolder(sModelFolder)
    #------------------------------------------------------------------------------------
#==================================================================================================











#==================================================================================================
class ExperimentQueueSystem(object):
    
    #------------------------------------------------------------------------------------
    def __init__(self):
        #....................... |  Instance Attributes | ...............................
        self.BaseFolder      = BaseFolders.EXPERIMENTS_SYSTEM
        self.ToEvaluteFolder = Storage.JoinPath(self.BaseFolder      , "toevaluate")
        self.PendingFolder   = Storage.JoinPath(self.BaseFolder      , "pending")
        self.ArchiveFolder   = Storage.JoinPath(self.BaseFolder      , "archive")
        self.ErrorFolder     = Storage.JoinPath(self.PendingFolder   , "errors") 
        self.EditFolder      = Storage.JoinPath(self.BaseFolder      , "edit")
        self.RecombineFolder = Storage.JoinPath(self.BaseFolder      , "recombine")

        
        self.CountersFileName   = Storage.JoinPath(self.BaseFolder, "counters")
        
        self.TemplateName           = None
        self.TemplateConfigFileName = None 
        #................................................................................
        Storage.EnsurePathExists(self.BaseFolder)
        Storage.EnsurePathExists(self.PendingFolder)
        Storage.EnsurePathExists(self.ArchiveFolder)
        Storage.EnsurePathExists(self.ErrorFolder)
        Storage.EnsurePathExists(self.EditFolder)
        Storage.EnsurePathExists(self.ToEvaluteFolder)
        Storage.EnsurePathExists(self.RecombineFolder)
        
        self.SetTemplateName(None)
    #------------------------------------------------------------------------------------
    def LoopExecution(self, p_sExecutableName):
        bContinue = True
        while bContinue and (self.GetNextConfig() is not None):
            oConfig = self.LoadNextConfig()
            print("[TALOS]: Next experiment in queue:%s" % oConfig.FileName)    
            nResult = Exec.Python(p_sExecutableName, [oConfig.Architecture, oConfig.DataSetName, str(oConfig.FoldNumber)])
            bContinue = (nResult == 0)
            if not bContinue:
                print("[TALOS]: Error occured:%s" % oConfig.FileName)
                self.ArchiveConfigAsError(oConfig.FileName)
    #------------------------------------------------------------------------------------
    def SetTemplateName(self, p_sName):
        if p_sName is not None:
            self.TemplateName = p_sName
        else:
            self.TemplateName = "template.cfg"
    
        self.TemplateConfigFileName = Storage.JoinPath(self.EditFolder, self.TemplateName)
    #------------------------------------------------------------------------------------
    def __sequenceLearningRate(self, p_sSourceFileName, p_nLearningRateSequence, p_nCounter):
        print("  -> Sequencing learning rates from template")
        sResult=[]   
        nCounter = p_nCounter
        for nLearningRate in p_nLearningRateSequence:
            oNewConfig = NNLearnConfig()
            oNewConfig.LoadFromFile(p_sSourceFileName)
            oNewConfig.Learn.LearningRate = nLearningRate
            _, sName, _ = Storage.SplitFileName(p_sSourceFileName)
            sDestFileName = Storage.JoinFileName(self.PendingFolder, "%.3d-" % p_nCounter + sName + "-lr%.6f" % nLearningRate, ".cfg")
            sResult.append(sDestFileName)
            oNewConfig.SaveToFile(sDestFileName)
            nCounter += 1
            
        return nCounter, sResult   
    #------------------------------------------------------------------------------------
    def __sequenceFoldNumber(self, p_sSourceFileName, p_nFoldSequence, p_nCounter):
        print("  -> Sequencing fold numbers from template")
        sResult=[]
        nCounter = p_nCounter
        for nFoldNumber in p_nFoldSequence:
            oNewConfig = NNLearnConfig()
            oNewConfig.LoadFromFile(p_sSourceFileName)
            oNewConfig.FoldNumber = nFoldNumber
            _, sName, _ = Storage.SplitFileName(p_sSourceFileName)
            sDestFileName = Storage.JoinFileName(self.PendingFolder, "%.3d-" % p_nCounter + sName + "-fold%d" % nFoldNumber, ".cfg")
            sResult.append(sDestFileName)
            oNewConfig.SaveToFile(sDestFileName)
            nCounter += 1
            
        return nCounter, sResult
    #------------------------------------------------------------------------------------
    def __readCounter(self):
        """ Gets the current run/evaluation counter """
        self.Counter = Storage.DeserializeObjectFromFile(self.CountersFileName)    
        if self.Counter is None:
            self.Counter = {"FormatVersion":"TALOS10", "RunCounter": 1}
            nCounter = 1
        else: 
            nCounter = self.Counter["RunCounter"]
            
        return nCounter           
    #------------------------------------------------------------------------------------
    def __writeCounter(self, p_nNumber):
        self.Counter["RunCounter"] = p_nNumber
        Storage.SerializeObjectToFile(self.CountersFileName, self.Counter, True)        
    #------------------------------------------------------------------------------------
    def IncCounter(self):
        nCounter = self.__readCounter()
        nCounter += 1
        self.__writeCounter(nCounter)
        return nCounter
    #------------------------------------------------------------------------------------
    def AddConfig(self, p_sConfigFileName=None):
        if p_sConfigFileName is None:
            sSourceFileName = self.TemplateConfigFileName
        else:
            sSourceFileName = p_sConfigFileName
        _, sName, _ = Storage.SplitFileName(sSourceFileName)
                    
        # Gets the current run/evaluation counter
        nCounter = self.__readCounter()    
                    
        oConfig = NNLearnConfig()
        oConfig.LoadFromFile(sSourceFileName)
        
        sDestFileNames = None
        if oConfig.LearningRateSequence is not None:
            nCounter, sDestFileNames = self.__sequenceLearningRate(sSourceFileName, oConfig.LearningRateSequence, nCounter)
        elif oConfig.FoldSequence is not None:
            nCounter, sDestFileNames = self.__sequenceFoldNumber(sSourceFileName, oConfig.FoldSequence, nCounter)
#             for nFoldNumber in oConfig.FoldSequence:
#                 oNewConfig = NNLearnConfig()
#                 oNewConfig.LoadFromFile(sSourceFileName)
#                 oNewConfig.FoldNumber = nFoldNumber
#                 sDestFileName = Storage.JoinFileName(self.PendingFolder, "%.3d-" % nCounter + sName, ".cfg")
#                 oNewConfig.SaveToFile(sDestFileName)
#                 nCounter += 1
        else:
            sDestFileNames = [Storage.JoinFileName(self.PendingFolder, "%.3d-" % nCounter + sName , ".cfg")]
            Storage.CopyFile(sSourceFileName, sDestFileNames[0])
            nCounter += 1
            
        # Saves the current run/evaluation counter
        self.__writeCounter()
        
        return sDestFileNames
    
    #------------------------------------------------------------------------------------
    def GetNextConfig(self):
        # By priority first evaluates models to save disk space and then start training 
        sResult = self.GetNextConfigToEvaluate()

        if sResult is None:
            sFiles = Storage.GetFilesSorted(self.PendingFolder)
            sConfigFiles = []
            for sFile in sFiles:
                _, _, sExt = Storage.SplitFileName(sFile)
                if sExt == ".cfg":
                    sConfigFiles.append(Storage.JoinPath(self.PendingFolder, sFile))
                    
            if len(sFiles) > 0:
                sResult = sConfigFiles[0]
            else:
                sResult = None
        
        return sResult
    #------------------------------------------------------------------------------------
    def GetNextConfigToEvaluate(self):
        sFiles = Storage.GetFilesSorted(self.ToEvaluteFolder)
        sConfigFiles = []
        for sFile in sFiles:
            _, _, sExt = Storage.SplitFileName(sFile)
            if sExt == ".cfg":
                sConfigFiles.append(Storage.JoinPath(self.ToEvaluteFolder, sFile))
                
        if len(sFiles) > 0:
            sResult = sConfigFiles[0]
        else:
            sResult = None
        
        return sResult
    #------------------------------------------------------------------------------------
    def EnsureConfig(self, p_sTemplateName):
        # Gets the next configuration, or copies from the current template file
        sConfigFileName = self.GetNextConfig()
        if sConfigFileName is None:
            # Sets the current configuration template file
            self.SetTemplateName(p_sTemplateName)
            self.AddConfig()
    #------------------------------------------------------------------------------------
    def LoadNextConfig(self):
        oConfig = None
        sNextConfigFileName = self.GetNextConfig()
        if sNextConfigFileName is not None:
            oConfig = NNLearnConfig()
            oConfig.LoadFromFile(sNextConfigFileName)
            
            # Supports the evaluation queue
            if sNextConfigFileName == self.GetNextConfigToEvaluate():
                oConfig.ParseUID()
                oConfig.IsTraining          = False
                oConfig.IsEvaluating        = True
                oConfig.IsDeterminingBest   = True
                
            
        return oConfig            
    #------------------------------------------------------------------------------------
    def ArchiveConfig(self, p_sConfigFileName=None, p_sDestFileName=None):
        if p_sConfigFileName is None:
            p_sConfigFileName = self.GetNextConfig()
            
        
        Storage.MoveFileToFolder(p_sConfigFileName, self.ArchiveFolder, p_sDestFileName)
    #------------------------------------------------------------------------------------
    def ArchiveConfigAsError(self, p_sConfigFileName=None):
        if p_sConfigFileName is None:
            p_sConfigFileName = self.GetNextConfig()
        Storage.MoveFileToFolder(p_sConfigFileName, self.ErrorFolder)        
    #------------------------------------------------------------------------------------

#==================================================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
#==================================================================================================
class ExperimentFolder(object):

    __verboseLevel = 2
    #------------------------------------------------------------------------------------
    @classmethod
    def SplitExperimentCode(cls, p_sExperimentCode):
        sParts = p_sExperimentCode.split("/")
        return int(sParts[0]), sParts[1]
    #------------------------------------------------------------------------------------
    @classmethod
    def GetExperimentName(cls, p_sModelName, p_sDataSetName=None):
        if p_sDataSetName is None:
            sResult = p_sModelName
        else:
            sResult = p_sDataSetName + "-" + p_sModelName
        return sResult        
    #------------------------------------------------------------------------------------
    @classmethod
    def GetLastRunConfig(cls, p_sModelName, p_sDataSetName, p_nFoldNumber):
        oExp = ExperimentFolder(p_sModelName, p_sDataSetName)
        oExp.__setFoldNumber(p_nFoldNumber)
        oExp.RunSub.Initialize(oExp.RunSub.LastUID, p_bIsEnsuringPaths=False)
        return oExp.RunSub.LearnConfigUsedFileName
    #------------------------------------------------------------------------------------
    @classmethod
    def GetExperiment(cls, p_sFolder, p_sCustomBaseFolder=None):
        oResult = None
        oConfig = NNLearnConfig.GetConfig(p_sFolder)
        if oConfig is not None:
            oResult = ExperimentFolder(p_oLearnConfig=oConfig)
            if p_sCustomBaseFolder is not None:
                oResult.RunBaseFolder = p_sCustomBaseFolder
            assert oConfig.SavedExperimentUID is not None
            oResult.Open(oConfig.FoldNumber, oConfig.SavedExperimentUID, p_bIsVerbose=False)
            oResult.LearnConfig = oConfig
        return oResult
    #------------------------------------------------------------------------------------        
        
        
        
    #------------------------------------------------------------------------------------
    def __init__(self, p_sModelName=None, p_sDataSetName=None, p_nBatchSize=15, p_oLearnConfig=None):
        #........ |  Instance Attributes | ..............................................
        self.ERL = None
        self.LearnConfig = p_oLearnConfig
        if self.LearnConfig is not None:
            p_sModelName = p_oLearnConfig.Architecture
            p_sDataSetName = p_oLearnConfig.DataSetName
            p_nBatchSize = p_oLearnConfig.BatchSize
        
        self.ModelName  = p_sModelName
        self.DataSetName = p_sDataSetName
        self.Name = ExperimentFolder.GetExperimentName(self.ModelName, self.DataSetName)
        self.Code = None
#         if self.DataSetName is None:
#             self.Name = p_sModelName
#         else :
#             self.Name = self.DataSetName + "-" + p_sModelName
 
        self.BatchSize = p_nBatchSize

        assert self.Name is not None, "Experiment name is not provided"
        
        
        self.BaseFolder = os.path.join(BaseFolders.EXPERIMENTS_STORE, self.Name)
        self.RunBaseFolder = os.path.join(BaseFolders.EXPERIMENTS_RUN, self.Name)
        
        # // Control Variables \\
        self.RandomSeed = 2017
        self.FoldNumber = None
        self.MaxDiskSpaceForModels = 30 #GB

                
        # // Composite Objects \\
        self.Log = Logger()
        self.MinuteUID=MinuteUID()
        self.IsNew = True
        self.StoreSub=None
        self.RunSub=None

        # // System \\
        self.OSSignature        = OperatingSystemSignature()
        self.TensorflowVersion  = tf.VERSION

                
        #................................................................................
        #Storage.EnsurePathExists(self.BaseFolder)
        #Storage.EnsurePathExists(self.RunBaseFolder)
    #------------------------------------------------------------------------------------
    @property
    def UID(self):
        return self.MinuteUID.UID
    #------------------------------------------------------------------------------------
    @UID.setter
    def UID(self, p_UID):
        self.MinuteUID.UID = p_UID
    #------------------------------------------------------------------------------------            
    def GetCode(self):
        return "%d/%s" % (self.FoldNumber, self.MinuteUID.UID)
    #------------------------------------------------------------------------------------
    def GetERLString(self):
        sERLString = "%s:%s|%d/%s" % (  self.DataSetName,
                                        self.ModelName,
                                        self.FoldNumber,
                                        self.MinuteUID.UID )
        return sERLString 
    #------------------------------------------------------------------------------------
    def GetDataSetFolder(self, p_sSubFolder):
        return Storage.JoinPath(BaseFolders.DATASETS, p_sSubFolder)
    #------------------------------------------------------------------------------------
    def __setFoldNumber(self, p_nFoldNumber):
        self.FoldNumber = p_nFoldNumber
        self.StoreSub   = ExperimentSubFolder(self, self.FoldNumber, False)
        self.RunSub    =  ExperimentSubFolder(self, self.FoldNumber, True)
    #------------------------------------------------------------------------------------
    def __determineInitialModelUID(self):
        sFiles = Storage.GetFilesSorted(self.RunSub.ArchitectureCommonFolder)
        sUID = None
        for sFile in sFiles:
            if sFile.startswith("initial-model_"):
                _, sName, _ = Storage.SplitFileName(sFile)
                sUID = sName[-12:]
        # A standard fold number 1 and the last saved initial experiment in the common folder will be returned
        return sUID
    #------------------------------------------------------------------------------------
    def GetInitialExperiment(self):
        assert self.LearnConfig is not None, "Method requires a learn configuration."
        oInitialExperiment=None

        if self.LearnConfig.InitExperimentCode is not None:
            if self.LearnConfig.InitExperimentCode == "=":
                sInitExperimentUID = self.__determineInitialModelUID()
            else:
                sInitExperimentUID = self.LearnConfig.InitExperimentUID

            # If automatic initial experiment is used, initial experiment UID is none for the first experiment.                
            if sInitExperimentUID is not None:
                oInitialExperiment = ExperimentFolder(self.LearnConfig.Architecture, self.LearnConfig.DataSetName, self.LearnConfig.BatchSize)
                oInitialExperiment.Open(self.LearnConfig.InitExperimentFoldNumber, sInitExperimentUID)
            
        return oInitialExperiment
    #------------------------------------------------------------------------------------
    def Activate(self):
        """ Returns
                True : If a new experiment folder is created and the configuration was copied there
                False: If an existing experiment folder is reused 
        """ 
        assert self.LearnConfig is not None, "Method requires a learn configuration."
        
        if self.LearnConfig.SavedExperimentUID is not None:
            self.Open(self.LearnConfig.FoldNumber, self.LearnConfig.SavedExperimentUID)
            bMustArchive = False
        else:
            if self.LearnConfig.IsTraining:
                self.Begin()
                # Copies the source configuration file to the experiment subfolder "config"
                Storage.CopyFile(self.LearnConfig.FileName, self.RunSub.LearnConfigFileName, True)
                bMustArchive = True
            else:
                self.Open()
                bMustArchive = False
            
        return bMustArchive 
    #------------------------------------------------------------------------------------
    def OpenERL(self, p_sERL=None, p_sERLString=None):
        if p_sERLString is not None:
            self.ERL = ERLString(p_sERLString)
        elif p_sERL is not None:
            self.ERL = p_sERL
        elif self.LearnConfig is not None:
            self.ERL = self.LearnConfig.ERL
        
        assert self.ERL is not None, "No ERL given"
        assert self.ERL.IsValid, "Invalid ERL %s" % self.ERL.String
        self.Open(self.ERL.FoldNumber, self.ERL.ExperimentUID)
    #------------------------------------------------------------------------------------
    def Open(self, p_nFoldNumber=None, p_UID=None, p_bIsVerbose=True):
        if (p_nFoldNumber is None) and (self.LearnConfig is not None):
            p_nFoldNumber = self.LearnConfig.FoldNumber
        assert p_nFoldNumber is not None
        
        self.__setFoldNumber(p_nFoldNumber)        

#         if self.RunSub.LastUID == ExperimentSubFolder.NO_SUBFOLDERS:
#             # If no experiment has been run for the fold number it starts the first experiment
#             self.Begin(p_nFoldNumber)
#         else:
        # Initializes with the given UID or the UID of the last experiment
        
        self.IsNew = False        
        if p_UID is None:
            self.MinuteUID.UID = self.RunSub.LastUID
        else:
            self.MinuteUID.UID = p_UID

        self.StoreSub.Initialize(self.MinuteUID.UID, p_bIsEnsuringPaths=False)
        self.RunSub.Initialize(self.MinuteUID.UID, p_bIsEnsuringPaths=False)
        
        if p_bIsVerbose:
            if type(self).__verboseLevel >= 1:
                print("[TALOS] Loaded experiment [%s], stored in %s, started at %s" % 
                        (  self.MinuteUID.UID, self.RunSub.ExperimentFolder
                         , self.MinuteUID.DateTime.strftime(tcc.LOG_DATETIME_FORMAT)  )
                      )
                
            self.Log.Open("",  p_nCustomLogFileName=self.RunSub.LogFileName, p_nLogType=Logger.LOG_TYPE_CONFIG)
            
        self.Code = "%d" % self.FoldNumber + "/" + self.MinuteUID.UID
        
        if self.ERL is None:
            self.ERL = ERLString(self.GetERLString())
    #------------------------------------------------------------------------------------
    def Begin(self, p_nFoldNumber=None):
        if (p_nFoldNumber is None) and (self.LearnConfig is not None):
            p_nFoldNumber = self.LearnConfig.FoldNumber
        assert p_nFoldNumber is not None
        
        self.__setFoldNumber(p_nFoldNumber)
        
        # Sets seeds for reproducibility
        random.seed(self.RandomSeed)
        np.random.seed(self.RandomSeed)
        
        # Initializes a new UID for the current minute and ensures the subfolders
        if not self.IsNew:
            self.MinuteUID = MinuteUID()
        self.StoreSub.Initialize(self.MinuteUID.UID, p_bIsEnsuringPaths=False)
        self.RunSub.Initialize(self.MinuteUID.UID)
        self.Code = "%d" % self.FoldNumber + "/" + self.MinuteUID.UID
        
        if type(self).__verboseLevel >= 2:
            print("Begin experiment at", self.MinuteUID.DateTime)
         
        self.Log.Open("",  p_nCustomLogFileName=self.RunSub.LogFileName, p_nLogType=Logger.LOG_TYPE_CONFIG)
        self.Log.WriteLine("Initializing experiment [%s] ..." % self.Code)
        self.Log.Flush()
    #------------------------------------------------------------------------------------
    def StoreCompressedModels(self):
        sZipFiles = self.RunSub.ListCompressedModels()
        sDestFolder = self.StoreSub.ExperimentModelFolder
        Storage.EnsurePathExists(sDestFolder)
        
        for sZipFile in sZipFiles:
            self.Log.Print("Moving model %s to storage folder %s" % (sZipFile, sDestFolder))
            Storage.MoveFileToFolder(sZipFile, sDestFolder)   

        Storage.DeleteEmptyFolder(self.RunSub.ExperimentModelFolder)     
    #------------------------------------------------------------------------------------
    def WriteGraph(self, p_oMainGraph):
        NetworkGraphWriter(self.RunSub.ExperimentConfigFolder, p_oMainGraph).Write()
    #------------------------------------------------------------------------------------
    def End(self):
        self.Log.Flush()
    #------------------------------------------------------------------------------------
#==================================================================================================    
    

    
    
    
                   
                        
                            

#==================================================================================================        
class NetworkGraphWriter(object):
    #------------------------------------------------------------------------------------    
    def __init__(self, p_sTensorboardLogFolder, p_oGraph):
        #........ |  Instance Attributes | ..............................................
        self.TensorboardLogFolder = p_sTensorboardLogFolder
        self.Graph = p_oGraph
        #................................................................................        
    #------------------------------------------------------------------------------------
    def Write(self):
        oSummaryWriter = tf.summary.FileWriter(logdir=self.TensorboardLogFolder, graph=self.Graph, filename_suffix=".tf") 
        oSummaryWriter.flush()
        
        oGraphDef = self.Graph.as_graph_def()
        
        sGraphDefStr = str(oGraphDef)
        with open(os.path.join(self.TensorboardLogFolder,"tensorflow-graph.txt"), "w") as oFile: 
            oFile.write(sGraphDefStr)
               
        self.__writeBatchFile(self.TensorboardLogFolder)
    #------------------------------------------------------------------------------------
    def __writeBatchFile(self, p_sFolder):
        sWinScriptFileName = os.path.join(p_sFolder,"browse-graph.bat")
        sLinuxScriptFileName =  os.path.join(p_sFolder,"bgr.sh")

        for nIndex, sFile in enumerate( [sWinScriptFileName, sLinuxScriptFileName] ):     
            with open(sFile, "w") as oFile:
                if nIndex == 0:
                    print("call %s %s" % (SysParams.AnacondaActivation, SysParams.AnacondaEnvironment), file=oFile)
                else:
                    print("#! /bin/bash", file=oFile)
                    print("source activate %s" % SysParams.AnacondaEnvironment, file=oFile)
                print("cd " + self.TensorboardLogFolder, file=oFile)
                print("tensorboard --logdir .", file=oFile)
    #------------------------------------------------------------------------------------               
#==================================================================================================    
    












    
#==================================================================================================    
class ClassificationEvaluator(object):    
    
    #------------------------------------------------------------------------------------
    def __init__(self, p_oNetwork, p_bIsDiscardingModels=True):
        #........ |  Instance Attributes | ..............................................
        self.Network = p_oNetwork
        
        self.ExperimentSub = self.Network.Experiment.RunSub        
        self.Models=None
        self.ModelIndex=None
        
        self.CurrentModelEpochNumber = None
        self.CurrentModelFolder = None
        self.HasLoadedModel = None
        self.IsDiscardingModels = p_bIsDiscardingModels
        self.IsAutoDeterminingBestModels = False
        #................................................................................
    #------------------------------------------------------------------------------------
    def FirstModel(self):
        self.Models = self.ExperimentSub.ListSavedModels()
        # If models folder is delete then checks the save results
        if self.Models == []:
            self.Models = self.ExperimentSub.ListSavedResults()
        
        self.ModelIndex = -1
        return self.NextModel() 
    #------------------------------------------------------------------------------------
    def NextModel(self):
        self.CurrentModelEpochNumber = None
        self.CurrentModelFolder = None
        self.ModelIndex += 1
        
        if self.ModelIndex < len(self.Models):
            oRec = self.Models[self.ModelIndex]
            self.CurrentModelEpochNumber = oRec[0]
            self.CurrentModelFolder      = oRec[1]
            
            if not self.IsExistingModelResults():
                self.Network.Log.Print("Evaluating folder %s" % self.CurrentModelFolder)
                self.Network.State.LoadWeights(self.CurrentModelEpochNumber)
                self.HasLoadedModel = True
            else:
                self.Network.Log.Print("Results found for epoch %d" % self.CurrentModelEpochNumber)
                self.HasLoadedModel = False
        
        return self.CurrentModelEpochNumber
    #------------------------------------------------------------------------------------
    def EndOfModels(self):
        return self.ModelIndex >= len(self.Models)
    #------------------------------------------------------------------------------------
    def IsExistingModelResults(self):
        #print(self.ExperimentSub.ModelResultsFileNameTemplate % self.CurrentModelEpochNumber)
        return Storage.IsExistingFile(self.ExperimentSub.ModelResultsFileNameTemplate % self.CurrentModelEpochNumber)
    #------------------------------------------------------------------------------------
    def CalculateMetrics(self, p_oPrediction):
        oMetrics = ClassificationMetrics()
        oMetrics.Calculate(p_oPrediction.Actual, p_oPrediction.Predicted, p_oPrediction.PredictedProbs)
        oMetrics.CalculateTopK(p_oPrediction.TopKappa, p_oPrediction.TopKCorrect)
        oMetrics.IDs = p_oPrediction.SampleIDs
        oMetrics.Save( self.ExperimentSub.ModelResultsFileNameTemplate % self.CurrentModelEpochNumber )
    #------------------------------------------------------------------------------------
    def DetermineBestModels(self, p_oExperiment=None):
        oBest = ClassificationBest(self.ExperimentSub.ExperimentResultsFolder)
        oBest.Run()
        oBest.Save(self.ExperimentSub.BestModelResultsFileName)
        oBest.ExportToText(self.ExperimentSub.BestModelTextFileName, p_oExperiment)
        
        # Discards all models except the best ones
        if self.IsDiscardingModels:
            self.ExperimentSub.DiscardModels(oBest.DiscardedEpochs)
            
        # Compresses the each model parameters folder into a zip file    
        self.ExperimentSub.CompressModels(oBest.BestEpochs)
        # Moves the model zip files to the neural network experiment storage
        self.Network.Experiment.StoreCompressedModels()
    #------------------------------------------------------------------------------------
    def Evaluate(self, p_oIterator):
        bMustStart = False
        self.FirstModel()
        while not self.EndOfModels():
            if self.HasLoadedModel:
                bMustStart = True
                break        
            self.NextModel()
        
        if bMustStart:
            p_oIterator.Start()
        
            self.FirstModel()
            while not self.EndOfModels():
                if self.HasLoadedModel:
                    # Recalls the data through the trained model
                    oPrediction = self.Network.PredictEval(p_oIterator)
                    self.CalculateMetrics(oPrediction)
                    p_oIterator.Resume()
                self.NextModel()
            
            p_oIterator.Stop(True)
            
        if self.IsAutoDeterminingBestModels:
            self.DetermineBestModels() 
    #------------------------------------------------------------------------------------
#==================================================================================================    
    
    
    
    
    
    
    
    
    
    










