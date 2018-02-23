# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        UTILITIES
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import sys
import platform


#[TO BE REFACTORED] #TEMPORARY VALUES
MODEL_DEFINITION_FOLDER = "F:\\NNets\\"


#------------------------------------------------------------------------------------
def IsPython35():
    return (sys.version_info.major==3) and (sys.version_info.minor==5)
#------------------------------------------------------------------------------------
def IsPython35Windowsx64(bIsCheckingForWindows81=False):
    # Returns if running on Windows 8.1 x64 and Python 3.5
    bIsWindows = (platform.architecture()[1] == "WindowsPE") 
    bIsWindows81 = (platform.version().startswith("6.3"))
    bIs64bit = platform.architecture()[0].startswith("64")
    
    Result = IsPython35() and bIsWindows and bIs64bit
    if bIsCheckingForWindows81:
        Result = Result and bIsWindows81 

    return Result    
#------------------------------------------------------------------------------------
def OperatingSystemSignature():
    sOSName = (platform.architecture()[1]) 
    sOSVersion = (platform.version())
    sOSBits = platform.architecture()[0]
    return sOSName + " " + sOSVersion + " " + sOSBits
#------------------------------------------------------------------------------------
def GetValue(p_dDict, p_sKey):
    if p_sKey in p_dDict:
        oValue = p_dDict[p_sKey]
        print("  | %s:" % p_sKey.ljust(32), oValue )
        return oValue
    else:
        return None
#------------------------------------------------------------------------------------
def GetValueAsBool(p_dDict, p_sKey):
    if p_sKey in p_dDict:
        oValue = p_dDict[p_sKey]
        print("  | %s:" % p_sKey.ljust(32), (oValue!= 0))
        return oValue != 0    
    else:
        return None
#------------------------------------------------------------------------------------    


