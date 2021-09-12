
from glob import glob  # For listing files
import cv2  # For handling image loading & processing
import os  # for path creation
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from skimage.registration import phase_cross_correlation
import scipy.ndimage as nd
#from scipy.optimize import minimize 
#import skimage as sk



def do_mosaic(INPUT_PATH='../../input/Video002_CLIP/images/', INPUT_PATH_SEG = '../../input_seg/Video002_CLIP/', OUTPUT_PATH = '../../Video002/'):
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH +' created')
    else:
        print(OUTPUT_PATH +' exists')
        
    input_file_list = sorted(glob(INPUT_PATH + "*.png"))
    
    powerv = 1
    thre = 10
    
    paint_size = 2048
    paint_window = np.zeros((paint_size,paint_size,3),dtype=np.float32)
    paint_window2 = np.zeros((paint_size,paint_size),dtype=np.float32)
    count_window = np.zeros((paint_size,paint_size),dtype=np.float32)
    
    pos = (0,0)
    
    
    use_paint_window = True
    use_txt_output = False
    
    
    for i in range(len(input_file_list)-1):
    
        img1_color =cv2.imread(input_file_list[i])
        img2_color =cv2.imread(input_file_list[i+1])
        img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        mask1 = img1>thre
        mask1 = nd.binary_erosion (mask1, iterations=5)
        mask2 = img2>thre     
        mask2 = nd.binary_erosion (mask2, iterations=5)         
        
        file_name = input_file_list[i].split("/")[-1]
        img1_seg = softmax(np.power(np.load(INPUT_PATH_SEG  + file_name.replace('png','npy')),powerv),axis=-1)[:,:,1]*255
        img1_seg = cv2.resize(img1_seg, img1.shape)
        
        file_name = input_file_list[i+1].split("/")[-1]        
        img2_seg = softmax(np.power(np.load(INPUT_PATH_SEG  + file_name.replace('png','npy')),powerv),axis=-1)[:,:,1]*255
        img2_seg = cv2.resize(img2_seg, img2.shape)    
         
        
        img1_org = img1_seg * mask1
        img2_org = img2_seg * mask2

        
        nz1 = np.nonzero(mask1)
        v1 = img1[nz1]
        min1 = v1.min(); 
        max1 = v1.max()
        

        nz2 = np.nonzero(mask2)
        v2 = img2[nz2]
        min2 = v2.min(); 
        max2 = v2.max()        
        
        minv = min(min1,min2)
        maxv = max(max1,max2)
        
        
        img1[nz1] = np.clip(((v1 - minv) / (maxv - minv) * 255),0,255).astype(np.uint8)
        img2[nz2] = np.clip(((v2 - minv) / (maxv - minv) * 255),0,255).astype(np.uint8)        
        
        
        img1 = img1_org
        img2 = img2_org
        
        
        shifted = phase_cross_correlation(img1, img2, reference_mask = mask1, overlap_ratio = 0.8)        
        
#        img2_shifted = nd.affine_transform(img2, np.eye(2), offset = -shifted)
#        mask2_shifted = nd.affine_transform(mask2, np.eye(2), offset = -shifted)
        
#        mat = get_matrix(img1, img2_shifted, mask1, mask2_shifted)
#        H = np.eye(3)
#        H[:2,:2] = mat
#        H[0:2,2] = -shifted
        
#        img2 = nd.affine_transform(img2, H,order=1)


        if (use_paint_window):
            if(i==0):
                pos = (paint_size//2-mask1.shape[0]//2,paint_size//2-mask1.shape[1]//2)
                paint_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1],:] += img1_color*mask1[:,:,np.newaxis]
                paint_window2[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += img1*mask1            
                count_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += mask1 
                
                
            pos = pos + shifted.astype(np.int32)
            paint_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1],:] += img2_color*mask2[:,:,np.newaxis]
            paint_window2[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += img2*mask2        
            count_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += mask2 
            ary = (paint_window2 / np.clip(count_window,a_min=1,a_max=100000)).astype(np.uint8)        
            plt.imshow(ary,cmap='gray')
            plt.show()        
            plt.imsave(OUTPUT_PATH+str(i).zfill(4)+'.png',ary)

        
        print(shifted, file_name)


        if (use_txt_output):    
            file_name = input_file_list[i+1].split("/")[-1]
            file_name = file_name.replace('png','txt')
            result = np.savetxt(OUTPUT_PATH  + file_name , "test",'%10.5f')


do_mosaic()





