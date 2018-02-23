# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        OBJECT WRAPPERS FOR TENSORS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================

import tensorflow as tf


#==================================================================================================
class TONumber(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_nStartValue, p_sScopeName, p_tDType=tf.float32, p_sName="value"):
        #........ |  Instance Attributes | ..............................................
        self.StartValue = p_nStartValue
        self.ScopeName = p_sScopeName
        self.DType = p_tDType
        self.__session = None
        self.__value = None
        with tf.variable_scope(self.ScopeName):
            self.Tensor   = tf.Variable(p_nStartValue, trainable=False, dtype=self.DType, name=p_sName)
        #................................................................................            
    #------------------------------------------------------------------------------------
    def ResetToStartValue(self):
        self.__value = self.StartValue
        self.Session.run(self.Tensor.assign(self.__value))
        return self.__value        
    #------------------------------------------------------------------------------------
    def MultiplyBy(self, p_nNumber):
        self.__value = self.Tensor.eval(session=self.Session)
        self.__value = self.__value * p_nNumber
        self.Session.run(self.Tensor.assign(self.__value))
    #------------------------------------------------------------------------------------
    def IncBy(self, p_nNumber):
        self.__value = self.Tensor.eval(session=self.Session)
        self.__value = self.__value + p_nNumber
        self.Session.run(self.Tensor.assign(self.__value))
    #------------------------------------------------------------------------------------
    @property
    def Session(self):
        return self.__session
    #------------------------------------------------------------------------------------
    @Session.setter
    def Session(self, p_oSession):
        self.__session = p_oSession
    #------------------------------------------------------------------------------------
    @property
    def Value(self):
        self.__value = self.Tensor.eval(session=self.Session)
        return self.__value
    #------------------------------------------------------------------------------------    
    @Value.setter
    def Value(self, p_oValue):
        self.__value = p_oValue
        self.Session.run(self.Tensor.assign(p_oValue))
    #------------------------------------------------------------------------------------
#==================================================================================================













#==================================================================================================
class TOBoolean(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_sScopeName, p_sDefault=False):
        #........ |  Instance Attributes | ..............................................
        self.ScopeName = p_sScopeName
        self.Tensor=None
        self.SetTrue=None
        self.SetFalse=None    
        self.Default=p_sDefault    
        self.Session=None
        #................................................................................            
        with tf.variable_scope(self.ScopeName):
            #tValue = tf.get_variable(self.Name, dtype=tf.bool, shape=[],initializer=tf.constant_initializer(self.Default),trainable=False)
            tValue = tf.Variable(self.Default, trainable=False, dtype=tf.bool, name="flag")
            self.Tensor   = tValue
            self.SetTrue  = tf.assign(tValue, True)
            self.SetFalse = tf.assign(tValue, False)            
    #------------------------------------------------------------------------------------
    def Set(self, p_oValue, p_oSession=None):
        if p_oSession is None:
            p_oSession = self.Session
            
        if p_oValue==True:
            self.SetTrue.eval(session=p_oSession)
        else:
            self.SetFalse.eval(session=p_oSession)
    #------------------------------------------------------------------------------------
    def Get(self, p_oSession=None):
        if p_oSession is None:
            p_oSession = self.Session
        
        Result = self.Tensor.eval(session=p_oSession)
        return Result
    #------------------------------------------------------------------------------------
#==================================================================================================