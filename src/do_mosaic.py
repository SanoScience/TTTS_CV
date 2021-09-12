"""
Fetoscopy placental vessel segmentation and registration challenge (FetReg)
EndoVis - FetReg2021 - MICCAI2021
Challenge link: https://www.synapse.org/#!Synapse:syn25313156

Task 2 - Registration - Docker dummy example showing 
the input and output folders and the output text file format for the submission

"""

import sys  # For reading command line arguments
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


#INPUT_PATH = sys.argv[1]
#OUTPUT_PATH = sys.argv[2]

INPUT_PATH = 'input/Video002_CLIP/images/'
INPUT_PATH_SEG = 'input_seg/Video002_CLIP/'
OUTPUT_PATH = 'Video002/'


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



if __name__ == "__main__":
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH +' created')
    else:
        print(OUTPUT_PATH +' exists')
        
    input_file_list = sorted(glob(INPUT_PATH + "/*.png"))
    

    kernel = np.ones((3, 3), np.uint8)
    thre = 10
    powerv = 1
    
    paint_size = 2048
    paint_window = np.zeros((paint_size,paint_size,3),dtype=np.float32)
    paint_window2 = np.zeros((paint_size,paint_size),dtype=np.float32)
    count_window = np.zeros((paint_size,paint_size),dtype=np.float32)
    
    pos = (0,0)
    
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

#        img1 = np.power(img1,powerv)
#        img2 = np.power(img2,powerv)  
        
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
        
#        minv = minv + 0.3 * (maxv-minv)
        
        img1[nz1] = np.clip(((v1 - minv) / (maxv - minv) * 255),0,255).astype(np.uint8)
        img2[nz2] = np.clip(((v2 - minv) / (maxv - minv) * 255),0,255).astype(np.uint8)        
        
        
        img1 = img1_org
        img2 = img2_org
        
        
        shifted = phase_cross_correlation(img1, img2, reference_mask = mask1, overlap_ratio = 0.8)#moving_mask = mask2)        
        
#        img2_shifted = nd.affine_transform(img2, np.eye(2), offset = -shifted)
#        mask2_shifted = nd.affine_transform(mask2, np.eye(2), offset = -shifted)
        
#        mat = get_matrix(img1, img2_shifted, mask1, mask2_shifted)
#        H = np.eye(3)
#        H[:2,:2] = mat
#        H[0:2,2] = -shifted
        
#        img2 = nd.affine_transform(img2, H,order=1)


        if(i==0):
            pos = (paint_size//2-mask1.shape[0]//2,paint_size//2-mask1.shape[1]//2)
            paint_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1],:] += img1_color*mask1[:,:,np.newaxis]
            paint_window2[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += img1*mask1            
            count_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += mask1 
            
            
        pos = pos + shifted.astype(np.int32)
        paint_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1],:] += img2_color*mask2[:,:,np.newaxis]
        paint_window2[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += img2*mask2        
        count_window[pos[0]:pos[0]+mask1.shape[0],pos[1]:pos[1]+mask1.shape[1]] += mask2 
        
        print(shifted, file_name)
#        ary = (paint_window / np.clip(count_window,a_min=1,a_max=100000)[:,:,np.newaxis]).astype(np.uint8)
        ary = (paint_window2 / np.clip(count_window,a_min=1,a_max=100000)).astype(np.uint8)        
        plt.imshow(ary)
        plt.show()        
        plt.imsave(OUTPUT_PATH+str(i).zfill(4)+'.png',ary)

        continue
    
        file_name = input_file_list[i+1].split("/")[-1]
        file_name = file_name.replace('png','txt')
        result = np.savetxt(OUTPUT_PATH  + file_name , "test",'%10.5f')











#        img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
#         img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
#         img1 = np.power(img1,4)
#         img2 = np.power(img2,4)
#         img1 = img1/img1.max()*255
#         img2 = img2/img2.max()*255        
#         img1 = img1.astype(np.uint8)
#         img2 = img2.astype(np.uint8)
        
# #        height, width = img2.shape
 
#         # Create ORB detector with 5000 features.
#         #        orb_detector = cv2.ORB_create(5000)
#         sift = cv2.xfeatures2d.SIFT_create()
 
#         # Find keypoints and descriptors.
#         # The first arg is the image, second arg is the mask
#         #  (which is not required in this case).
#         #kp1, d1 = orb_detector.detectAndCompute(img1, None)
#         #kp2, d2 = orb_detector.detectAndCompute(img2, None)
#         kp1, d1 = sift.detectAndCompute(img1, None)
#         kp2, d2 = sift.detectAndCompute(img2, None)

#         kp_img = cv2.drawKeypoints(img1_color, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         plt.imshow(kp_img)
#         plt.show()

#         continue
 
#         # Match features between the two images.
#         # We create a Brute Force matcher with
#         # Hamming distance as measurement mode.
#         matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
 
#         # Match the two sets of descriptors.
#         matches = matcher.match(d1, d2)
 
#         # Sort matches on the basis of their Hamming distance.
#         matches.sort(key = lambda x: x.distance)
 
#         # Take the top 90 % matches forward.
#         matches = matches[:int(len(matches)*0.9)]
#         no_of_matches = len(matches)
 
#         # Define empty matrices of shape no_of_matches * 2.
#         p1 = np.zeros((no_of_matches, 2))
#         p2 = np.zeros((no_of_matches, 2))
 
#         for i in range(len(matches)):
#             p1[i, :] = kp1[matches[i].queryIdx].pt
#             p2[i, :] = kp2[matches[i].trainIdx].pt
 
#         # Find the homography matrix.
#         homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)









# kp, des = sift.detectAndCompute(gray_img, None)

# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(gray_img, None)

# kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('SIFT', kp_img)
# cv2.waitKey()


# Initiate ORB detector
#orb = cv.ORB_create()
# find the keypoints with ORB
#kp = orb.detect(img,None)
# compute the descriptors with ORB
#kp, des = orb.compute(img, kp)


        # crop_lim = [crop_y,img1.shape[0]-crop_y,crop_x,img1.shape[1]-crop_x]
        # img1 = img1[crop_lim[0]:crop_lim[1],crop_lim[2]:crop_lim[3], ]
        # img2 = img2[crop_lim[0]:crop_lim[1],crop_lim[2]:crop_lim[3]]
        # plt.imshow(img1)
        # plt.show()
    
    
        # ShiftReg    = cv2.reg_MapperGradShift()
        # PyrShiftReg = cv2.reg_MapperPyramid(ShiftReg)
        # ShiftMap    = PyrShiftReg.calculate(img1, img2)
        # ShiftMap    = cv2.reg.MapTypeCaster_toShift(ShiftMap)
    
        # AffineMap   = cv2.reg_MapAffine(np.eye(2),ShiftMap.getShift())
    
        # AffineReg    = cv2.reg_MapperGradAffine()
        # PyrAffineReg = cv2.reg_MapperPyramid(AffineReg)
        # AffineMap    = PyrAffineReg.calculate(img1, img2,AffineMap)
        # AffineMap    = cv2.reg.MapTypeCaster_toAffine(AffineMap)
    
        # H = np.eye(3)
        # H[0:2,2:3] = ShiftMap.getShift()
        # ProjMap = cv2.reg_MapProjec(H)
    
        # ProjReg    = cv2.reg_MapperGradProj()
        # PyrProjReg = cv2.reg_MapperPyramid(ProjReg)
        # ProjMap    = PyrProjReg.calculate(img1, img2,ProjMap)
        # ProjMap    = cv2.reg.MapTypeCaster_toProjec(ProjMap)        

        # import cv2
        # import numpy as np
         
        # # Open the image files.
        # img1_color = cv2.imread("align.jpg")  # Image to be aligned.
        # img2_color = cv2.imread("ref.jpg")    # Reference image.
         
        # # Convert to grayscale.
        # img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
        # height, width = img2.shape
         
        # # Create ORB detector with 5000 features.
        # orb_detector = cv2.ORB_create(5000)
         
        # # Find keypoints and descriptors.
        # # The first arg is the image, second arg is the mask
        # #  (which is not required in this case).
        # kp1, d1 = orb_detector.detectAndCompute(img1, None)
        # kp2, d2 = orb_detector.detectAndCompute(img2, None)
         
        # # Match features between the two images.
        # # We create a Brute Force matcher with
        # # Hamming distance as measurement mode.
        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
         
        # # Match the two sets of descriptors.
        # matches = matcher.match(d1, d2)
         
        # # Sort matches on the basis of their Hamming distance.
        # matches.sort(key = lambda x: x.distance)
         
        # # Take the top 90 % matches forward.
        # matches = matches[:int(len(matches)*0.9)]
        # no_of_matches = len(matches)
         
        # # Define empty matrices of shape no_of_matches * 2.
        # p1 = np.zeros((no_of_matches, 2))
        # p2 = np.zeros((no_of_matches, 2))
         
        # for i in range(len(matches)):
        #   p1[i, :] = kp1[matches[i].queryIdx].pt
        #   p2[i, :] = kp2[matches[i].trainIdx].pt
         
        # # Find the homography matrix.
        # homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
         
        # # Use this matrix to transform the
        # # colored image wrt the reference image.
        # transformed_img = cv2.warpPerspective(img1_color,
        #                     homography, (width, height))
         
        # # Save the output.
        # cv2.imwrite('output.jpg', transformed_img)