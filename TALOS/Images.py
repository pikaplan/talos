# =====================================================================================================================
#        
#        Tensor Abstraction Layer Objects 0.0.8-ALPHA
#        IMAGE PROCESSING FUNCTIONS
#
#        Framework design by Pantelis I. Kaplanoglou
#        Licensed under the MIT License
#
# =====================================================================================================================
import numpy as np
from matplotlib import colors
from PIL import Image, ImageChops, ImageFilter
from TALOS.Core import phi


 
#------------------------------------------------------------------------------------
#Analyse the image to H,S,B,L
def AnalyzeImageToHSBL(p_oImg):
    LUMA_W = np.asarray([0.29889531/255.0, 0.58662247/255.0, 0.11448223/255.0], dtype=np.float32).T
    
    oImg = p_oImg/255.0
    img_hsv     =colors.rgb_to_hsv(oImg)
    hue         =img_hsv[:,:,0]
    saturation  =img_hsv[:,:,1]
    #brightness  =img_hsv[:,:,2]
    luma        = np.dot(p_oImg,LUMA_W)
    return hue,saturation,luma 

#------------------------------------------------------------------------------------
def ConvertImageRGBToHIF(p_oImg):
    img_hue,img_saturation,img_luma=AnalyzeImageToHSBL(p_oImg)
    p_oHIFImg=np.dstack((img_hue.astype(np.float32),img_saturation.astype(np.float32),img_luma.astype(np.float32)))
    return p_oHIFImg
#------------------------------------------------------------------------------------
def LoadImageFromFile(p_sFileName):
    img = Image.open(p_sFileName).convert('RGB')
    return np.array(img)    
    
#------------------------------------------------------------------------------------
def LoadImageAndCropToSize(p_sFileName, p_tSize=(227,227)):
    img = Image.open(p_sFileName).convert('RGB')
    img_width = float(img.size[0])
    img_height = float(img.size[1])
    
    nAspectRatio = img_width / img_height
    
    
        
    if img_width > img_height:
        ratio =  img_width/img_height  
        new_width=int(p_tSize[0]*ratio)
        new_height=p_tSize[0]
    else:
        ratio = img_height/img_width
        new_width=p_tSize[0]
        new_height=int(p_tSize[0]*ratio)
    
    img = img.resize((new_width, new_height), Image.NONE)
    
    half_the_width = img.size[0] // 2
    half_the_height = img.size[1] // 2
    
    nLeftDiff = p_tSize[0] // 2
    nTopDiff = p_tSize[1] // 2
    nRightDiff = nLeftDiff
    nBottomDiff = nTopDiff
    
    if (2 * nLeftDiff) != p_tSize:
        nRightDiff += 1 
    if (2 * nTopDiff) != p_tSize:
        nBottomDiff += 1  
    
    img_cropped = img.crop(
        (
            half_the_width - nLeftDiff,
            half_the_height - nTopDiff,
            half_the_width + nRightDiff,
            half_the_height + nBottomDiff
        )
    )
    
    return np.array(img_cropped)
    
#------------------------------------------------------------------------------------
def LoadImageAndFitToSize(p_sFileName, p_tSize=(227,227)):
    img = Image.open(p_sFileName).convert('RGB')
    img_width = float(img.size[0])
    img_height = float(img.size[1])
    
    if img_width > img_height:
        ratio = img_height / img_width
        new_width=p_tSize[0]
        new_height=int(p_tSize[0]*ratio)
    else:
        ratio = img_width / img_height
        new_width=int(p_tSize[0]*ratio)
        new_height=p_tSize[0]
    
    img = img.resize((new_width, new_height), Image.NONE)
    
    thumb = img.crop( (0, 0, p_tSize[0], p_tSize[1]) )

    offset_x = max( (p_tSize[0] - img.size[0]) / 2, 0 )
    offset_y = max( (p_tSize[1] - img.size[1]) / 2, 0 )

    img = ImageChops.offset(thumb, int(offset_x), int(offset_y))
    
        
    return np.array(img)
#------------------------------------------------------------------------------------
def RotateInWhiteBackground(p_nPILImage, p_nDegrees):
    # original image
    img = p_nPILImage
    # converted to have an alpha layer
    im2 = img.convert('RGBA')
    # rotated image
    rot = im2.rotate(p_nDegrees)#, expand=1)
    # a white image same size as rotated image
    fff = Image.new('RGBA', rot.size, (255,)*4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(rot, fff, rot).convert(img.mode)
    # save your work (converting back to mode='1' or whatever..)
    return out

#------------------------------------------------------------------------------------
def LoadImageAndMakeAugmentedSquare(p_sFileName, p_tSize=(227,227)):
    nSize = p_tSize
    
    nHalfSize   = (nSize[0] // 2, nSize[1] // 2)
    nModX       = nSize[0] % 2
    nModY       = nSize[1] % 2

    img = Image.open(p_sFileName).convert('RGB')
    
    img_width = float(img.size[0])
    img_height = float(img.size[1])
    nAspectRatio = img_width / img_height
    if nAspectRatio > 1.0:
        bIrregular = nAspectRatio > (phi*0.9)
        bIsTopBottomPadding = True
    else:
        bIrregular = nAspectRatio < (1.0/(phi*0.9))
        bIsTopBottomPadding = False
    
    #print("[%d,%d]  AspectRatio:%.4f  Irregular:%r" % (img_width, img_height, nAspectRatio, bIrregular))
            
    nRatioWidth = 1.0
    nRatioHeight = 1.0        
    if bIrregular:
        if img_width > img_height:
            nRatioHeight = img_height/img_width
        else:
            nRatioWidth = img_width/img_height
    else:            
        if img_width > img_height:       
            nRatioWidth = img_width/img_height 
        else:
            nRatioHeight = img_height/img_width      
                            
    new_width=int(nSize[0]*nRatioWidth)
    new_height=int(nSize[1]*nRatioHeight)
    
    img = img.resize((new_width, new_height), Image.NONE)
    #print("New Image Size", img.size)
    
            
    if bIrregular:
        thumb = img.crop( (0, 0, nSize[0], nSize[1]) )
    
        offset_x = int( max( (nSize[0] - img.size[0]) / 2, 0 ) )
        offset_y = int( max( (nSize[1] - img.size[1]) / 2, 0 ) )
    
        img = ImageChops.offset(thumb, offset_x, offset_y)
    
        Result = np.array(img)
           
        
        #TODO: Fadding out by number of space size
        if bIsTopBottomPadding:
            space_size_top=offset_y
            space_size_bottom=nSize[1]-new_height-offset_y
            #print("top %i, bottom %i" %(space_size_top, space_size_bottom))
            
            first_row=Result[offset_y+1,:,:]
            last_row=Result[offset_y+new_height-1,:,:]
            #first_row=np.repeat( np.mean(first_row, axis=0).reshape(1, Result.shape[2]),  Result.shape[1], axis=0) 
            #last_row=np.repeat( np.mean(first_row, axis=0).reshape(1, Result.shape[2]),  Result.shape[1], axis=0 )
    
            top_rows=np.repeat(first_row.reshape(1, Result.shape[1], Result.shape[2]), space_size_top+1, axis=0)
            bottom_rows=np.repeat(last_row.reshape(1, Result.shape[1], Result.shape[2]), space_size_bottom, axis=0)
            
            im1 = Image.fromarray(top_rows)
            im1 = im1.filter(ImageFilter.BLUR)
            top_rows = np.array(im1)
                    
            im2 = Image.fromarray(bottom_rows)
            im2 = im2.filter(ImageFilter.BLUR)
            bottom_rows = np.array(im2)        
            
            
            Result[0:offset_y+1,:,:]=top_rows[:,:,:]
            Result[offset_y+new_height:nSize[1],:,:]=bottom_rows[:,:,:]
        else:        
            
            space_size_left=offset_x
            space_size_right=nSize[0]-new_width-space_size_left
            #print("left %i, right %i" %(space_size_left, space_size_left)) 
                
                
            first_col=Result[:,offset_x+1,:]
            last_col=Result[:,offset_x+new_width-1,:]
    
            left_cols=np.repeat(first_col.reshape(Result.shape[0], 1, Result.shape[2]), space_size_left+1, axis=1)
            right_cols=np.repeat(last_col.reshape(Result.shape[0], 1, Result.shape[2]), space_size_right, axis=1)
    
            
            im1 = Image.fromarray(left_cols)
            im1 = im1.filter(ImageFilter.BLUR)
            left_cols = np.array(im1)
                    
            im2 = Image.fromarray(right_cols)
            im2 = im2.filter(ImageFilter.BLUR)
            right_cols = np.array(im2)  
            
    
            Result[:,0:offset_x+1,:]=left_cols[:,:,:]
            Result[:,offset_x+new_width:nSize[0],:]=right_cols[:,:,:]
        
        img = Image.fromarray(Result)       
    
    #print("Base Image Size", img.size)
    #plt.imshow(np.array(img))
    #plt.show()       
            
    if nAspectRatio > 1.0:
        nDiff = (img.size[0] - img.size[1]) // 2
    else:
        nDiff = (img.size[1] - img.size[0]) // 2
#     
#     
#     if False:
#         a4im = Image.new('RGB',
#                          (595, 842),   # A4 at 72dpi
#                          (255, 255, 255))  # White
#         a4im.paste(img, img.getbbox())  # Not centered, top-left corne
#         plt.imshow(np.array(a4im))
#         plt.show()
    nCenterX = img.size[0] // 2
    nCenterY = img.size[1] // 2
    
    nImgCropped=[None]*3
    if nDiff > 40:
        nCropPositions = [0, -nDiff//2, nDiff//2]
    else:
        nCropPositions = [0]
    
    for nIndex,nShiftPos in enumerate(nCropPositions):
        nPosX = nCenterX
        nPosY = nCenterY
        if nAspectRatio > 1.0:
            nPosX += nShiftPos
        else:
            nPosY += nShiftPos
        
        nLeft   = nPosX - nHalfSize[0]
        nRight  = nPosX + nHalfSize[0] + nModX
        
        nTop    = nPosY - nHalfSize[1]
        nBottom = nPosY + nHalfSize[1] + nModY
        nImgCropped[nIndex] = np.array(img.crop( (nLeft, nTop, nRight, nBottom) ))
        
        
    if len(nCropPositions) == 1:
        nImgCropped[1] = np.array(RotateInWhiteBackground(img, 12))
        nImgCropped[2] = np.array(RotateInWhiteBackground(img, -12))
    
    return nImgCropped         
#     for nIndex, nImg in enumerate(nImgCropped):
#         if nImg is not None:
#             print(nIndex, nImg.shape)  
#             plt.imshow(nImg)
#             plt.show()
#     


