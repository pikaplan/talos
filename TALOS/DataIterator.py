# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.1.0-ALPHA
#        DATA ITERATORS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import time
import threading
import numpy as np
from TALOS.FileSystem import Storage



#==================================================================================================
class MLDataPager(object):

    
    __verboseLevel = 1
    #------------------------------------------------------------------------------------
    def __init__(self, p_sDataFolder, p_nTotalPageCount, p_bIsRecalling=False, p_bIsWarmup=False, p_nValidationPageStart=0, p_nValidationPageCount=0):
        #........ |  Instance Attributes | ..............................................
        self.DataFolder = p_sDataFolder
        self.TotalPageCount   = p_nTotalPageCount
        self.IsRecalling = p_bIsRecalling
        self.IsWarmup  = p_bIsWarmup
        self.IsTesting = (p_nValidationPageCount - p_nValidationPageStart) == 0
        
        self.ValidationPageCount  = p_nValidationPageCount
        self.ValidationPageStart  = p_nValidationPageStart
        self.ValidationPageEnd    = self.ValidationPageStart + self.ValidationPageCount
        self.ValidationPercentage = self.ValidationPageCount / p_nTotalPageCount
        
        self.CurrentPageStart   = -2        
        self.PageIndex          = 0
        self.Page               = [None]*4
        self.IDs                = None        
        self.Samples            = None
        self.Targets            = None
        self.HasData            = False
        self.PageNumbers        = None
        self.MaxPageIndex       = 0        
        self.Flags              = None
        self.BatchSize          = None
        #................................................................................
        self.__preparePageNumbers()
        
    #------------------------------------------------------------------------------------
    def __preparePageNumbers(self):
        self.WarmupPages=[]
        self.PageNumbers=[]
        for nPage in np.arange(0, self.TotalPageCount, dtype=np.int32):
            if self.IsRecalling:
                if self.IsTesting:
                    self.PageNumbers.append(nPage)
                elif (nPage >= self.ValidationPageStart) and (nPage < self.ValidationPageEnd):
                    self.PageNumbers.append(nPage)   
            else:
                if (nPage < self.ValidationPageStart) or (nPage >= self.ValidationPageEnd):
                    self.PageNumbers.append(nPage) 
                    if self.IsWarmup:
                        if len(self.WarmupPages) < self.ValidationPageCount:
                            self.WarmupPages.append(nPage)
        
        if self.IsWarmup:
            self.PageNumbers = self.WarmupPages
        
        self.PageNumbers = np.asarray(self.PageNumbers, dtype=np.int32)         
        
        if not self.IsRecalling:     
            self.PageNumbers = np.random.permutation(self.PageNumbers)
            
        self.MaxPageIndex = len(self.PageNumbers)
        if type(self).__verboseLevel >= 2:
            if self.IsRecalling:
                sCount = "   [@] Recall Pages:%d " % len(self.PageNumbers)
            else:
                sCount = "   [@] Training Pages  :%d " % len(self.PageNumbers)
            print(sCount, self.PageNumbers)
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
    def _filterOutSampled(self):
        nNewIDs     = np.zeros( self.IDs.shape     , dtype=np.int32)
        nNewSamples = np.zeros( self.Samples.shape , dtype=self.Samples.dtype)
        nNewTargets = np.zeros( self.Targets.shape , dtype=np.int32)
        
        nFlags = self.Flags[self.IDs]
        
        # Leaves out the samples that have been flagged
        nPos=0
        for nIndex, nMargin in enumerate(nFlags):
            if nMargin == 0:
                nNewIDs[nPos]           = self.IDs[nIndex]
                nNewSamples[nPos,:,:,:] = self.Samples[nIndex,:,:,:]
                nNewTargets[nPos]       = self.Targets[nIndex]
                nPos += 1
        
        # Completes the last batch with excluded samples to 
        nIndex = 0
        nExtra = 0
        while (nPos % self.BatchSize) != 0:
            nMargin = nFlags[nIndex]
            if nMargin != 0:
                nNewIDs[nPos]           = self.IDs[nIndex]
                nNewSamples[nPos,:,:,:] = self.Samples[nIndex,:,:,:]
                nNewTargets[nPos]       = self.Targets[nIndex]
                nPos += 1
                nExtra += 1
            nIndex += 1
            
        if type(self).__verboseLevel >= 2:
            print("Page was filtered. Total:%s Extra:%s" % (nPos, nExtra))
        
        assert self.IDs.shape[0] > 0, "IDs are zero" 
        
        self.IDs     = nNewIDs[0:nPos]
        self.Samples = nNewSamples[0:nPos]
        self.Targets = nNewTargets[0:nPos]
    #------------------------------------------------------------------------------------
    def SetActiveDataPages(self, p_nCurrentPageStart):
        oData1 = self.Page[p_nCurrentPageStart]     
        oData2 = self.Page[p_nCurrentPageStart + 1]
        
        if oData2 is None:
            if self.IsTesting:
                self.IDs = np.asarray(oData1["IDs"]) 
                self.Samples = np.asarray(oData1["Samples"])
                self.Targets = np.expand_dims(oData1["Targets"], axis=1)               
            else:
                self.IDs, self.Samples, self.Targets = self.__shuffleArrays(
                     np.asarray(oData1["IDs"]) 
                    ,np.asarray(oData1["Samples"])
                    ,np.expand_dims(oData1["Targets"], axis=1)
                    )    
        else:
            if self.IsTesting:
                self.IDs = np.concatenate( [ oData1["IDs"],oData2["IDs"] ], axis=0) 
                self.Samples = np.concatenate( [ oData1["Samples"],oData2["Samples"] ], axis=0)
                self.Targets = np.expand_dims(np.concatenate( [ oData1["Targets"],oData2["Targets"] ], axis=0), axis=1)    
            else:             
                self.IDs, self.Samples, self.Targets = self.__shuffleArrays(
                     np.concatenate( [ oData1["IDs"],oData2["IDs"] ], axis=0)
                    ,np.concatenate( [ oData1["Samples"],oData2["Samples"] ], axis=0)
                    ,np.expand_dims(np.concatenate( [ oData1["Targets"],oData2["Targets"] ], axis=0), axis=1)
                    )    
            
        # Restructure the data based excluding flagges samples, before concatenation
        if self.Flags is not None:
            self._filterOutSampled()
    #------------------------------------------------------------------------------------
    def LoadFromDisk(self):
        self.CurrentPageStart += 2
        if self.CurrentPageStart > 2:
            self.CurrentPageStart = 0
            
        sFileName1 = Storage.JoinPath(self.DataFolder, MLDataIterator.FILENAME_TEMPLATE_PAGE  % (self.PageNumbers[self.PageIndex]))
        oData1 = Storage.DeserializeObjectFromFile(sFileName1, p_bIsVerbose=False)
        self.Page[self.CurrentPageStart]     = oData1
        if type(self).__verboseLevel >= 2:
            print("   [>] Load MEM%d: %d" % (self.CurrentPageStart, self.PageNumbers[self.PageIndex]))

        if self.PageIndex + 1 < len(self.PageNumbers):         
            sFileName2 = Storage.JoinPath(self.DataFolder, MLDataIterator.FILENAME_TEMPLATE_PAGE % (self.PageNumbers[self.PageIndex + 1]))
            oData2 = Storage.DeserializeObjectFromFile(sFileName2, p_bIsVerbose=False)
            if type(self).__verboseLevel >= 2:
                print("   [>] Load MEM%d: %d " % (self.CurrentPageStart + 1, self.PageNumbers[self.PageIndex + 1]))
        else:
            oData2 = None
        self.Page[self.CurrentPageStart + 1] = oData2    
    #------------------------------------------------------------------------------------
    def __iter__(self):
        self.LoadFromDisk()
        return self
    #------------------------------------------------------------------------------------
    def __next__(self):
        nActivePageStart = self.CurrentPageStart
        nActivePageIndex = self.PageIndex
        
        'Returns the next value till current is lower than high'
        if self.PageIndex < self.MaxPageIndex:
            # Sets the current data pages that other threads may read
            self.SetActiveDataPages(nActivePageStart)
            self.HasData = True
            
            
            # Returns a dictionary for the current data pages
            if nActivePageIndex+ 1 < len(self.PageNumbers):
                nPageNumbers = [self.PageNumbers[nActivePageIndex], self.PageNumbers[nActivePageIndex+ 1]]
                nActivePageEnd = nActivePageStart + 1
            else:
                nPageNumbers = [self.PageNumbers[nActivePageIndex], None]
                nActivePageEnd = None

            if type(self).__verboseLevel >= 2:
                print("   [>] Return MEM%d: %d" % (nActivePageStart, nPageNumbers[0]))
                if nActivePageEnd is not None:
                    print("   [>] Return MEM%d: %d" % (nActivePageEnd, nPageNumbers[1]))

            # Start reading the next pages from disk
            self.PageIndex += 2
            bFinished = self.PageIndex >= self.MaxPageIndex
            if not bFinished: 
                self.LoadFromDisk()
                
            oData = {
                "PageIndexStart" : nActivePageStart
               ,"PageIndexEnd"   : nActivePageEnd
               ,"PageNumbers"    : nPageNumbers
               ,"Finished"       : bFinished
               ,"IDs"            : self.IDs
               ,"Samples"        : self.Samples
               ,"Targets"        : self.Targets
            }
            return oData            
        else:
            raise StopIteration
    #------------------------------------------------------------------------------------    
#==================================================================================================
class MLDataIterator(object):
    __verboseLevel = 1
    
    
    FILENAME_TEMPLATE_PAGE = "page%.4d.npy"
    SLEEP_TIME = 0.05 #secs
    #------------------------------------------------------------------------------------
    def __init__(self, p_sDataFolder, p_nTotalSamples, p_nPageSize, p_bIsValidation=False, p_nFoldNumber=None, p_nFolds=10, p_nValidationPageStart=0, p_nValidationPageCount=0, p_sName=None, p_nBatchSize=15):
        #........ |  Instance Attributes | ..............................................
        self.DataFolder = Storage.JoinPath(p_sDataFolder, "")
        self.IsStarted = False
        self.IsFinished = False
        self.IsWaiting = False
        self.Continue = False
        self.Stopped = False
        self.FinishedCondition=None
        self.MustReadNextData=None
        self.Cycles=None
        self.TotalSamples = p_nTotalSamples
        self.PageSize = p_nPageSize
        self.TotalPageCount = self.TotalSamples / self.PageSize
        
        
        # Support for last page with less samples
        # TODO: Support training with additional validation set
        if p_nFoldNumber is None:
            self.TotalPageCount = np.ceil(self.TotalPageCount)
        else:
            assert self.TotalPageCount == int(self.TotalPageCount), "Count of pages must be an integer. Total Samples %d / PageSize %d = %f" % (self.TotalSamples, self.PageSize, self.TotalPageCount)
            assert self.TotalPageCount % 2 == 0, "Count of pages must be an even number"
        
        self.IsValidation = p_bIsValidation
        

        if p_nFoldNumber is None:
            self.FoldIndex = None
            self.Folds = None
            self.ValidationPageStart = p_nValidationPageStart
            self.ValidationPageCount = p_nValidationPageCount
        else:
            self.FoldIndex = p_nFoldNumber - 1
            self.Folds = p_nFolds
            self.ValidationPageCount = self.TotalPageCount / self.Folds
            assert self.ValidationPageCount == int(self.ValidationPageCount), "Count of validation pages must be an integer. TotalPageCount:%d  Folds:%s TotalSamples:%d self.PageSize:%d" % (self.TotalPageCount, self.Folds, self.TotalSamples, self.PageSize)
            self.ValidationPageStart = self.FoldIndex * self.ValidationPageCount
        
        self.ValidationPercentage=self.ValidationPageCount/self.TotalPageCount
        
        self.TotalValidationSamples = self.ValidationPageCount * self.PageSize 
        self.TotalTrainSamples = self.TotalSamples - self.TotalValidationSamples
        if p_bIsValidation:
            self.TotalIteratedSamples = self.TotalValidationSamples
            self.Name = "VAL"
        else:
            self.TotalIteratedSamples = self.TotalTrainSamples
            self.Name = "TRN"
            
        if p_sName is not None:
            self.Name = p_sName 

        
        
        self.SampleIndex=0
        self.TotalCachedSamples=0
        self.BatchSize=p_nBatchSize
        self.EpochSamples = 0        
        self.IsEpochFinished = False
        
        if p_nFoldNumber is None:
            self.IsRecalling = True
        else:
            self.IsRecalling = self.IsValidation
        
        self.__isWarmup         = None
        self.TotalBatches       = None
        self.ValidationBatches  = None
        self.TrainingBatches    = None
        
        self.IsWarmup           = False
        #self.__createDataPager()
        #self.__recalculateBatchCount()
        
        self.ValidationIterator = None
        if not self.IsValidation:
            if self.ValidationPercentage > 0:
                if type(self).__verboseLevel >= 2:
                    print("=|=\t[%s:MLDataIterator] Batches - Total:%d  Training:%d  Validation%d" % (self.Name, self.TotalBatches, self.ValidationBatches, self.TrainingBatches)) 
                self.ValidationIterator = MLDataIterator( self.DataFolder, self.TotalSamples, self.PageSize, True
                                                         , p_nValidationPageStart = self.ValidationPageStart
                                                         , p_nValidationPageCount = self.ValidationPageCount 
                                                         , p_nBatchSize = p_nBatchSize
                                                         )
            self.Flags = np.zeros([self.TotalSamples], np.float32)
        else:
            self.Flags = None
            
        self.__isFilteringOutSamples=False
        #................................................................................
    #------------------------------------------------------------------------------------
    @property
    def IsWarmup(self):
        return self.__isWarmup
    #------------------------------------------------------------------------------------
    @IsWarmup.setter
    def IsWarmup(self, p_nValue):
        self.__isWarmup = p_nValue
        self.Pager = None
        self.__createDataPager()
        self.__recalculateBatchCount()        
    #------------------------------------------------------------------------------------
    def __recalculateBatchCount(self):
        if self.__isWarmup:
            nTotalSamples = self.Pager.MaxPageIndex * self.PageSize
        else:
            nTotalSamples = self.TotalSamples
            
        self.TotalBatches = nTotalSamples / self.BatchSize
        
        # Support for last page with less samples
        if self.FoldIndex is None:
            nTotalSamples = np.ceil(nTotalSamples)
 
        assert  self.TotalBatches == int(self.TotalBatches), "Count of minbatches must be integer"
        self.TotalBatches = int(self.TotalBatches)
        
        if self.__isWarmup:
            self.ValidationBatches = 0
        else:
            self.ValidationBatches = int(self.TotalBatches * self.ValidationPercentage)
        self.TrainingBatches = self.TotalBatches - self.ValidationBatches
    #------------------------------------------------------------------------------------
    def __createDataPager(self):
        if type(self).__verboseLevel >= 3:
            print("----> Recreating the pager", self.__isWarmup)
        
        self.Pager = MLDataPager(self.DataFolder, self.TotalPageCount
                                 ,p_bIsRecalling=self.IsRecalling, p_bIsWarmup=self.__isWarmup
                                 ,p_nValidationPageStart=self.ValidationPageStart, p_nValidationPageCount=self.ValidationPageCount)
        self.Pager.BatchSize = self.BatchSize
    #------------------------------------------------------------------------------------
    def EnableFiltering(self):
        self.__isFilteringOutSamples = True
    #------------------------------------------------------------------------------------
    def DisableFiltering(self):
        self.__isFilteringOutSamples = False
    #------------------------------------------------------------------------------------        
    def WaitForData(self):
        ''' Wait until the worker loop has fetched data '''
        while not self.Pager.HasData:
            time.sleep(MLDataIterator.SLEEP_TIME)
    #------------------------------------------------------------------------------------
    def FirstBatch(self):
        self.WaitForData()
        
        self.TotalCachedSamples = self.Pager.Targets.shape[0]
        self.SampleIndex = -self.BatchSize
        return self.NextBatch()
    #------------------------------------------------------------------------------------
    def NextBatch(self):
        self.SampleIndex += self.BatchSize
        
        if self.EndOfData():
            nStart = None
            nEnd = None        
        else:
            nStart = self.SampleIndex
            nEnd = nStart + self.BatchSize
            if nEnd >= self.TotalCachedSamples:
                nEnd = self.TotalCachedSamples
                
            self.EpochSamples += (nEnd - nStart)
            if type(self).__verboseLevel >= 3:
                print("Epoch samples:%d  [%d, %d]" % (self.EpochSamples, nStart, nEnd))
        
        return nStart, nEnd
    #------------------------------------------------------------------------------------
    def EndOfData(self):
        return self.SampleIndex >= self.TotalCachedSamples
    #------------------------------------------------------------------------------------
    def GetIsEpochFinished(self):
#         if type(self).__verboseLevel >= 3:
#             print("IsEpochFinished", self.EpochSamples, self.TotalIteratedSamples)
#         bResult = self.EpochSamples >= self.TotalIteratedSamples

        if self.IsEpochFinished:
            if type(self).__verboseLevel >= 2:
                print("Epoch %d iterated samples %d" % (self.Cycles, self.EpochSamples))
            self.EpochSamples = 0
            
        return self.IsEpochFinished
    #------------------------------------------------------------------------------------
    def ThreadMain(self):
        ''' Main worker loop of the data iterator thread '''
        if type(self).__verboseLevel >= 2:
            print("=|=\t[%s:MLDataIterator] starting ..." % self.Name)
         
        self.IsStarted = True
        self.Cycles=1
        while self.Continue:
            if self.__isFilteringOutSamples:
                self.Pager.Flags = self.Flags
            else:
                self.Pager.Flags = None
                
            self.IsEpochFinished = False
            for oData in self.Pager:
#                 #TEMP
#                 self.IDs        = oData["Targets"]
#                 self.Samples    = oData["Samples"]
#                 self.Targets    = oData["Targets"]
#                 #nPageIndexStart=oData["PageIndexStart"]
#                 #nPageIndexEnd=oData["PageIndexEnd"]
                self.IsEpochFinished=oData["Finished"]
                
                self.MustReadNextData.acquire()
                self.IsWaiting = True
                self.MustReadNextData.wait()
                self.MustReadNextData.release()
                self.IsWaiting=False
            
            # Create iteration for the next epoch of data usage            
            if self.Continue:
                self.__createDataPager()
                self.Cycles += 1
                
    #print(nPageIndexStart, nPageIndexEnd, nPageNumbers, nTargets[0], nTargets[nPageSize*2 - 1], "Checksum:", np.sum(sCheck))


        self.IsFinished=True   
             
    #------------------------------------------------------------------------------------
    def Start(self):
        ''' Creates and starts the data iterator thread '''
        
        if self.IsStarted:
            self.Resume()
        else:
            self.EpochSamples = 0
            self.IsFinished = False
            self.Continue = True
            self.Stopped  = False
            self.Thread = threading.Thread(target=self.ThreadMain)
            
            self.FinishedCondition=threading.Condition()
            self.MustReadNextData=threading.Condition()
            self.Thread.start()
          
        return self.Thread
    #------------------------------------------------------------------------------------
    def Stop(self, p_bIsForcing=False):
        ''' Stops the data iterator thread '''

        if self.IsStarted:
            # Stop the worker loop
            self.Continue = False
            
            if p_bIsForcing:
                self.Pager.HasData = True
                
            #self.FinishedCondition.acquire()
            self.MustReadNextData.acquire()
    
            #self.FinishedCondition.wait()
            self.MustReadNextData.notify()
            self.MustReadNextData.release()
            
            self.Thread.join(5000) # 5secs
            
            if type(self).__verboseLevel >= 2:
                print("=|= main() gracefully joined with %s:MLDataIterator" % self.Name)
    #------------------------------------------------------------------------------------
    def Resume(self):
        ''' Resumes the worker thread that reads data pages from the disk '''
        self.Pager.HasData = False
                
        while not self.IsWaiting:
            time.sleep(MLDataIterator.SLEEP_TIME)
            
        if type(self).__verboseLevel >= 2:
            print("=|=\t%s:MLDataIterator resuming ..." % self.Name)

        self.MustReadNextData.acquire()
        self.MustReadNextData.notify()
        self.MustReadNextData.release()    
    #------------------------------------------------------------------------------------            
#==================================================================================================







        
        
        
        















