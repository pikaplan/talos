# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        LEARNING CURVE PLOTS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
from TALOS.FileSystem import Storage,BaseFolders
from TALOS.Visualization import MultiSerieGraph
from TALOS.Experiments import ExperimentFolder
from TALOS.TrainingStats import StatsColumnType
from TALOS.Core import JSONConfig



#==================================================================================================
class LearningComparisonSettings(JSONConfig):
    #------------------------------------------------------------------------------------
    def __init__(self, p_oConfig=None, p_sFileName=None):
        super(LearningComparisonSettings,self).__init__(p_sFileName)
        #........................ |  Instance Attributes | ..............................
        self.Config = p_oConfig
        self.Metrics = []
        self.Titles = []
        self.ExperimentsToCompare = []
        self.ExperimentDescriptions = []
        #................................................................................
        #self.ExperimentBaseFolder = Storage.JoinPath(BaseFolders.EXPERIMENTS_RUN
        #                           , ExperimentFolder.GetExperimentName(self.Config.Architecture, self.Config.DataSetName))
        if self.FileName is None:
            self.FileName = Storage.JoinPath(BaseFolders.EXPERIMENTS_RUN, "learn-comparison.cfg")
    #------------------------------------------------------------------------------------
    def SetDefaults(self):
        self.Metrics = [
          "ValAvgF1Score",
          "TrainAvgF1Score",
          "TrainStepAvgF1Score",
          "TrainAverageError",
          "ValError",
          "TrainAverageAccuracy",
          "ValAccuracy",
          "ValAccuracyPerError",
          "ValOverfitError",
          "ValOverfitAccuracy",
          "ValOverfitRate",
          "EpochTotalTime",
          "TwoEpochTotalTime"
        ]
        
        self.Titles = [
          "ValAvgF1",
          "TrainAvgF1",
          "TrainStepAvgF1",
          "TrainCCE",
          "ValCCE",
          "TrainAcc",
          "ValAcc",
          "APER",
          "LossOFR",
          "AccOFR",
          "OFR",
          "Time",
          "Time (per 2 Epochs)"
        ]
    #------------------------------------------------------------------------------------
    def CompareWithThis(self, p_oExperiment):
        # Lazy initialization of the list
        bMustAdd = False
        if self.ExperimentsToCompare is None:
            bMustAdd = True
        elif self.ExperimentsToCompare == []:
            bMustAdd = True
        if bMustAdd:
            self.ExperimentsToCompare = []
            self.ExperimentsToCompare.append(None)
            self.ExperimentDescriptions = []
            self.ExperimentDescriptions.append(None) 
        
        # Sets as first experiment in comparison list
        bIsInitial = False
        if len(self.ExperimentsToCompare) == 1:
            if self.ExperimentsToCompare[0] is None:
                bIsInitial = True
        if bIsInitial:
            # Sets the current experiment in comparison list
            self.ExperimentsToCompare[0]   = p_oExperiment.GetERLString()
            self.ExperimentDescriptions[0] = "initial"
        else:
            # Adds a second experiment  
            if len(self.ExperimentsToCompare) < 2:
                # Sets the current experiment in comparison list
                self.ExperimentsToCompare.append(None)
                self.ExperimentDescriptions.append(None)
             
            self.ExperimentsToCompare[-1] = p_oExperiment.GetERLString()
            self.ExperimentDescriptions[-1] = "this"
    #------------------------------------------------------------------------------------   
    def DoAfterLoad(self):
        self.Metrics                = self.Get("Metrics")
        self.Titles                 = self.Get("Titles")
        self.ExperimentsToCompare   = self.Get("ExperimentsToCompare")
        self.ExperimentDescriptions = self.Get("ExperimentDescriptions")
    #------------------------------------------------------------------------------------
    def DoBeforeSave(self):
        self.Set("Metrics"                  , self.Metrics)
        self.Set("Titles"                   , self.Titles)
        self.Set("ExperimentsToCompare"     , self.ExperimentsToCompare)
        self.Set("ExperimentDescriptions"   , self.ExperimentDescriptions)
    #------------------------------------------------------------------------------------    
#==================================================================================================    
    
    
    
    
    
    
    
    

#==================================================================================================
class LearningComparison(object):
    DEFAULT_SERIE_COLORS = [
         '#4dbeee' # light-blue
        ,'#77ac30' # green
        ,'#d95319' # orange            
        ,'#7e2f8e' # purple
        ,'#edb120' # yellow            
        ,'#0072bd' # blue            
        ,'#a2142f' # red              
        ];
                
    #------------------------------------------------------------------------------------
    def __init__(self, p_oSettings=None):
        #........................ |  Instance Attributes | ..............................
        self.Settings = p_oSettings
        self.Metrics = None
        self.SerieLabels = None
        self.ExperimentsToCompare = None
        self.Epochs = None
        self.Envs = []
        self.Stats =[]        
        #................................................................................
    #------------------------------------------------------------------------------------   
    def Initialize(self, p_sCustomBaseFolder=None):
        if self.Metrics is None:
            self.Metrics        = self.Settings.Metrics
            self.SerieLabels    = self.Settings.Titles
            
        if self.ExperimentsToCompare is None:
            self.ExperimentsToCompare = self.Settings.ExperimentsToCompare
            self.Epochs = np.zeros(len( self.ExperimentsToCompare) + 1, np.int32)
            
        self.ModelTitles=[]
        for nIndex, sExperimentERL in enumerate(self.ExperimentsToCompare):
            if p_sCustomBaseFolder is not None:
                # Here a subfolder is given and the custom base folder is prepended
                sExperimentFolder = Storage.JoinPath(p_sCustomBaseFolder, sExperimentERL)
                oExperiment = ExperimentFolder.GetExperiment(sExperimentFolder, p_sCustomBaseFolder)
                assert oExperiment is not None, "Experiment folder %s not found" % sExperimentFolder
                # Sets the config that is needed to return architecture and dataset for the learn comparison
                if self.Settings.Config is None:
                    self.Settings.Config = oExperiment.LearnConfig
                 
            else:
                oExperiment = ExperimentFolder(p_oLearnConfig=self.Settings.Config)
                oExperiment.OpenERL(p_sERLString=sExperimentERL)            
            #nFoldNumber, sUID = ExperimentFolder.SplitExperimentCode(oExperimentCode)
            #oExperiment = ExperimentFolder(p_oLearnConfig=self.Settings.Config)
            #oExperiment.Open(nFoldNumber, sUID)
            
            dStats = Storage.DeserializeObjectFromFile(oExperiment.RunSub.StatsFileName)
            assert dStats is not None, "File not found %s" % oExperiment.RunSub.StatsFileName
            
            self.Envs.append(oExperiment)
            self.Stats.append(dStats)
            self.Epochs[nIndex] = dStats["EpochNumber"] - 1
            #nFoldNumber, sUID = ExperimentFolder.SplitExperimentCode(oExperiment.Code)
            self.ModelTitles.append(self.Settings.ExperimentDescriptions[nIndex] + " (%s)" % oExperiment.ERL.ExperimentUID )
    #------------------------------------------------------------------------------------
    def Render(self):
        # Prepares the series with nulls
        nMaxEpochs = np.amax(self.Epochs)
        x    = np.arange(0, nMaxEpochs)
        
        sPlotFolder = self.Envs[-1].RunSub.ExperimentPlotFolder
         
        print("[>] Epochs in different models:%s  Maximum: %s" % (self.Epochs, nMaxEpochs))
        print(" |__ Ploting to folder: %s" % sPlotFolder)      
    
        nMaxColumn = StatsColumnType.VALUE 
        
        for nColIndex in range(0, nMaxColumn + 1):
            for nMetricIndex, sMetric in enumerate(self.Metrics):
                y=[]  
                oLabels=[]  
                
                if True:
                    print("     |___ ", sMetric)
                    for nSerieIndex, dStats in enumerate(self.Stats):
                        nMaxOfY = self.Epochs[nSerieIndex]
                
                        if sMetric.startswith("Custom"):
                            nValError  = dStats["ValError"][:, 0][:nMaxOfY]
                            nTrainError = dStats["TrainAverageError"][:, 0][:nMaxOfY]
                            nYAll = (nValError -  nTrainError) / nValError 
                        else:
                            if sMetric in dStats:
                                nYAll = dStats[sMetric]
                            else:
                                nYAll = None
                                print("Warning: Metric %s not found in stats" % sMetric)
                        
                        if nYAll is not None:
                            if (sMetric != "ValAccuracyPerError") and (sMetric != "EpochTotalTime") \
                                 and (sMetric != "EpochRecallTime") and (not sMetric.startswith("Custom")):
                                nYSlice = nYAll[:,nColIndex][:nMaxOfY]
                            else:
                                nYSlice = nYAll[:][:nMaxOfY]
                                
                            nY = np.zeros(nMaxEpochs, np.float32)
                            nY[:] = None                    
                            #nY[:nMaxOfY]=nYSlice[:]
                            if sMetric == "EpochTotalTime":
                                print(dStats["EpochTotalTime"])
                            nY[:nMaxOfY]=nYSlice[:]
                            y.append(nY)
                            #oLabels.append(StatsColumnType.ToString(self.SerieLabels[nMetricIndex], nColIndex) + " (%s)" % self.ModelTitles[nSerieIndex])
                            oLabels.append(self.ModelTitles[nSerieIndex])
        
        
                
                sTitle      = "Comparison of CNN models on %s" % self.Settings.Config.DataSetName
                #sTitle      = "Training of BioCNNs with GLAVP layer"
                sCaptionX   = "Training Epoch"
                sCaptionY   = StatsColumnType.ToString(self.SerieLabels[nMetricIndex], nColIndex)
                
                oGraph = MultiSerieGraph()
                oGraph.Setup.LegendFontSize=10
                oGraph.Setup.Title              = sTitle
                oGraph.Setup.CaptionX           = sCaptionX
                oGraph.Setup.CaptionY           = sCaptionY
                oGraph.Setup.CommonLineWidth    = 1.5
                oGraph.Setup.DisplayFinalValue  = True
                
                oGraph.Initialize(x, y, p_oLabels=oLabels, p_oColors=type(self).DEFAULT_SERIE_COLORS)
                oGraph.Render()
        
                bPlot=True
                if (sMetric == "ValAccuracyPerError") and (nColIndex>0):
                    bPlot=False 
                if (sMetric.startswith("Custom")) and (nColIndex>0):
                    bPlot=False
                
                if bPlot:   
                    
                    oGraph.Plot( Storage.JoinPath(sPlotFolder, "%02i. %s-%i.png" % (nMetricIndex, self.SerieLabels[nMetricIndex],nColIndex)) )
                    #oGraph.Plot(ExperimentEnvironment.EXPERIMENTSPACE_FOLDER + "=Results=\\%02i. %s-%i.png" % (nMetricIndex, sSerieLabels[nMetricIndex],nColIndex))
    #------------------------------------------------------------------------------------
#==================================================================================================        