
from glob import glob  # For listing files
import cv2  # For handling image loading & processing
import os  # for path creation
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from skimage.registration import phase_cross_correlation
import scipy.ndimage as nd
from scipy.optimize import minimize 
import skimage as sk

# make movied out of pngs
# ffmpeg -r 20 -f image2 -s 2048x2048 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test2.mp4


def dist(m, img_target, img, mask):
    trans = nd.affine_transform(img, m.reshape((2,2,)), order=1)
    err = np.linalg.norm((img_target-trans)*mask)
    return err 

def get_matrix(img_target, img, mask_target, mask_moving, const = 0.05):

    # find largest 
    mask = []
    mask_tot = np.bitwise_not((mask_target + mask_moving) > 0)
    for rad in range((img.shape[0]+img.shape[1])//4, (img.shape[0]+img.shape[1])//16,-1):
        maski = sk.draw.circle(img.shape[0]/2, img.shape[1]/2, rad, shape=None)
        mask = np.zeros(img.shape)
        mask[maski]=1
        if np.sum(mask * mask_tot)==0:
            break
    # apply largest possible contruction
    rad = rad * (1-const) * (1-const) - const * const
    maski =  sk.draw.circle(img.shape[0]/2, img.shape[1]/2, rad, shape=None)
    mask = np.zeros(img.shape)
    mask[maski]=1

    initial = np.array([1,0,0,1])
    bounds = [[1-const,1+const],[0-const,0+const],[0-const,0+const],[1-const,1+const]]

    sol = minimize(dist, x0=initial, bounds=bounds, args=(img_target, img, mask))
    return sol.x.reshape((2,2))




def do_mosaic(INPUT_PATH='../../input/Video002_CLIP/images/', INPUT_PATH_SEG = '../../input_seg/Video002_CLIP/', OUTPUT_PATH = '../../output/Video002/'):
    
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
    
    
    use_paint_window = False
    use_txt_output = True
    
    
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
         
        
        img1_seg = img1_seg * mask1
        img2_seg = img2_seg * mask2
        
        img1_seg_org = img1_seg.copy()
        img2_seg_org = img2_seg.copy()

        
        nz1 = np.nonzero(mask1)
        v1 = img1_seg[nz1]
        min1 = v1.min() 
        max1 = v1.max()
        

        nz2 = np.nonzero(mask2)
        v2 = img2_seg[nz2]
        min2 = v2.min()
        max2 = v2.max()        
        
        minv = min(min1,min2)
        maxv = max(max1,max2)
        
        
        img1_seg[nz1] = np.clip(((v1 - minv) / (maxv - minv) * 255),0,255).astype(np.uint8)
        img2_seg[nz2] = np.clip(((v2 - minv) / (maxv - minv) * 255),0,255).astype(np.uint8)        
        
        

        
        
        shifted = phase_cross_correlation(img1_seg, img2_seg, reference_mask = mask1, overlap_ratio = 0.8)        
        
#        img2_seg_shifted = nd.affine_transform(img2_seg, np.eye(2), offset = -shifted)
#        mask2_shifted = nd.affine_transform(mask2, np.eye(2), offset = -shifted)
#        
#        mat = get_matrix(img1_seg, img2_seg_shifted, mask1, mask2_shifted)
#        H = np.eye(3)
#        H[:2,:2] = mat
#        H[0:2,2] = -shifted
        
#        img2_org = nd.affine_transform(img2_org, H,order=1)


        if (use_paint_window):
            if(i==0):
                pos = (paint_size//2-mask1.shape[0]//2,paint_size//2-mask1.shape[1]//2)
                paint_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1],:] += img1_color*mask1[:,:,np.newaxis]
                paint_window2[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += img1_seg_org*mask1            
                count_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += mask1 
                
                
            pos = pos + shifted.astype(np.int32)
            paint_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1],:] += img2_color*mask2[:,:,np.newaxis]
            paint_window2[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += img2_seg_org*mask2        
            count_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += mask2 
            ary = (paint_window / np.clip(count_window,a_min=1,a_max=100000)[:,:,np.newaxis]).astype(np.uint8)        
            plt.imshow(ary,cmap='gray')
            plt.show()        
            plt.imsave(OUTPUT_PATH+str(i).zfill(4)+'.png',ary, cmap='gray')

        
        print(shifted, file_name, '\n', shifted)


        if (use_txt_output):    
            HH = np.eye(3)
            HH[0,2]=shifted[1]
            HH[1,2]=shifted[0]
            file_name = input_file_list[i+1].split("/")[-1]
            file_name = file_name.replace('png','txt')
            np.savetxt(OUTPUT_PATH  + file_name , HH,'%10.5f')


do_mosaic()





