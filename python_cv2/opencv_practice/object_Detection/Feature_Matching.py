import numpy as np 
import cv2
import matplotlib.pyplot as plt 


def display(img,name,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    plt.title(str(name))
    plt.show()

def display2(img1,img2,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(211)
    ax1.imshow(img1,cmap='gray')
    ax2 = fig.add_subplot(212)
    ax2.imshow(img2,cmap='gray')
    plt.show()

reeses = cv2.imread('../opencv_practice/DATA/reeses_puffs.png',0)

#display(reeses)


cereals = cv2.imread('../opencv_practice/DATA/many_cereals.jpg',0)
display2(reeses,cereals)

#####################################################################

#####################
##### ORB Method ###
###################

# Build create object
orb = cv2.ORB_create()

# Find the key point and descriptors
kp1,des1 = orb.detectAndCompute(reeses,None)
kp2,des2 = orb.detectAndCompute(cereals,None)

# Create the matching object as 'Brute Force Matching'
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

matches = bf.match(des1,des2)

# Sort the value
single_match = matches[0]
#print (single_match.distance)

matches = sorted(matches,key=lambda x:x.distance)
#print (len(matches))

single_match = matches[0]
#print (single_match.distance)

# Draw the matches
reeses_matches = cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:30],None,flags=2)

display(reeses_matches,name='ORB Method')
#####################################################################


######################
##### SIFT Method ###
####################

# The opencv is not support xfeature2d upper 3.4x
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 = sift.detectAndCompute(reeses,None)
kp2,des2 = sift.detectAndCompute(cereals,None)

bf = cv2.BFMatcher()

# Find the k best feature
matches = bf.knnMatch(des1,des2,k=2)

# Less distance == Better match
# Ratio Test ---> Match1 < 75% Match2
good = []
for match1,match2 in matches:
    # If match 1 distance is less than 75% of match 2 distance
    # Then descriptor was a good match, Lets keep it.
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

print ('Length of good feature:',len(good))
print ('Length of all feature:',len(matches))

sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)

display (sift_matches,'SIFT Method')
#####################################################################

##############################
#####  Flann Based Method ###
############################

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(reeses,None)
kp2,des2 = sift.detectAndCompute(cereals,None)

# FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(check=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []
for match1,match2 in matches:
    if match1.distance < 0.7*match2.distance:
        good.append([match1])

flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)
display(flann_matches,'FLANN Method')

############################################################################

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# MatchesMask

# Create the mask
matchesMask = [[0,0] for i in range(len(matches))]

good = []
for i,(match1,match2) in enumerate(matches):
    if match1.distance < 0.7*match2.distance:
        matchesMask[i] = [1,0]

# Create drawing parameter 
draw_params = dict(matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=0)

flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)
display(flann_matches,'FLANN Method use mask')



