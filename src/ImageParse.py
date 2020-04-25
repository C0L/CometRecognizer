#------------------------------------------------------------------------------#
# ImageParse.py
# Cogs 108
# This defines the majority of the image parsing functionality that our program
# will need. All these methods are inported into Main.py and can be
# accessed through the ImageParse.methodName(), as this is a static class
# defining static variables, which is kind of strange in python but
# this seems like the best solution. 
#
# The cv2 package is for opencv, which alows for manipulation of image files in
# a more simple manner. It can be installed with "pip" package manager which is
# standard for python. Numpy is imported as np for its array functionality.
# xwlt is a package that may be useful for writting out to xls files. This is
# not neccesary but it will be nice if the program constructs a file for us
# already when we build the training set. This way we can more easily add in the
# olive moments to our training data.
#------------------------------------------------------------------------------#
import cv2                  # Image manipulation and transforms
import numpy as np          # Contiguous arrays for efficiency
import xlwt                 # write to xls files
import os                   # File manipulation and deletion
from xlwt import Workbook   # Interact with MS excel files

# Once a comet is identified, this is the region that will be scrapped from the
# image. We split the X into pos and neg because the tail always goes to the
# right so there is no point is scraping a uniform sized chunk. The actual
# region here is just an estimate and Mara we need to know how large this region
# actually should be.
RECT_X_POS = 120
RECT_X_NEG = 70
RECT_Y = 70

# Max distance between circles before we throw the data out 50
CDIST = 50
BLUR = (15,15)
HOUGHS = 4.9

#------------------------------------------------------------------------------#
# ImageParse
# This is a static class which provides all the image manipulation
# functionality. Specifics are left to the function headers.
#------------------------------------------------------------------------------#
class ImageParse():
    # imgNum is the count of images read, cNum is the number of comets found
    imgNum = 0
    cNum = 0

    #--------------------------------------------------------------------------#
    # serializeFile
    # This will take an input file, comb through each image in the file, extract
    # the comets from each image and screen for bad values, then write the good
    # comets and key to the output file.
    # @param  inputFile - File containing images to scrape comets from
    # @param  outFile   - Where to write the scraped comets and keys
    # @return None
    #--------------------------------------------------------------------------#
    def serializeFile(inputFile, outFile):
        # Start by gathering all of the files in the output directory
        delFiles = os.listdir(outFile)
        # Iterate through files in output dir and del to prevent buiildup
        for d in delFiles:
            os.remove(outFile + d)

        # Then get a list of all the input files
        files = os.listdir(inputFile)
        # Iterate through the files and call comet scraper on input file
        for f in files:
            ImageParse.serializeImage((inputFile + f), outFile)
        
        ImageParse.serializeXLS(outFile, ImageParse.cNum)

    #--------------------------------------------------------------------------#
    # serializeImage
    # This takes an individual image and gets all of the comets in the image.
    # Any bad comets are removed from the list of comets which will be written
    # out to file.
    # @param  inImage - The individual image that will be scraped for comets 
    # @param  outFile - Where to write the scraped comets and key to
    # @return None
    #--------------------------------------------------------------------------#
    def serializeImage(inImage, outFile):
        # Get the image object and locations of circles from getSplit
        circs, img = ImageParse.getSplit(inImage)

        # Get rid of all decimals to avoid any rounding errors
        circs = np.round(circs[0, :]).astype("int")

        # Only the circles that meet our criteria will be stored in this
        gCircs = [] 

        # Iterate through all of the circles detected to determine what to keep
        for circ in circs:
            # Check if the given circle is a bad value, or "sanatize"
            if ImageParse.sanitize(circ, circs, img):
                # Skip the rest of the process if this circle is a bad value
                continue

            # Get the region defined by RECT_X/Y to capture only a single comet
            split = img[(circ[1] - RECT_Y) : (circ[1] + RECT_Y), 
                        (circ[0] - RECT_X_NEG) : (circ[0] + RECT_X_POS)]
            # Write out the indivdual comet to our output dir
            cv2.imwrite(outFile + os.path.basename(inImage) 
                        + "_split_" + str(ImageParse.cNum) + ".tif", split)
            # Add this circle to our list of good circles
            gCircs.append(circ)
            # The number of images read is then increased by 1
            ImageParse.cNum += 1
        
        # After all circles processed then write out key with number comet label
        ImageParse.serializeKey(img, gCircs, inImage, outFile)

    #--------------------------------------------------------------------------#
    # serializeXLS
    # No current implementation
    # @param  path - Where to write xls
    # @param nCircs - number of circles that were written
    # @return None
    #--------------------------------------------------------------------------#
    def serializeXLS(path, nCircs):
        wb = Workbook()
        s1 = wb.add_sheet('#->OLIVEMOMENT')
        s1.write(0,0, '#')
        s1.write(0,1, 'moment')

        for i in range(nCircs):
            s1.write(i+1,0, str(i))

        wb.save(path + 'oliveMoments.xls')

    #--------------------------------------------------------------------------#
    # serializeKey
    # This will take the screened circles and then write out an identifier to
    # each of the good comets. This will assist with transcribing the olive
    # moments into our training data set.
    # @param img      - The original image which we write identifiers over 
    # @param circs    - The list of circles which were screened
    # @param inImage  - The file name so we can write with correct name
    # @param  outFile - Where to write the key to
    # @return None
    #--------------------------------------------------------------------------#
    def serializeKey(img, circs, inImage, outFile):
        # Reset img counter to be consistent with the xls file
        ImageParse.cNum -= len(circs)

        # Iterate through all of the comets to add identifier
        for (x, y, r) in circs:
            # Add a circle to encapsulate the comet
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            # Add a unique number to the comet which corresponds to image
            cv2.putText(img, str(ImageParse.cNum), (x, y), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # We have written out one more commet
            ImageParse.cNum += 1

        # Then write out the ket to the given input fille
        cv2.imwrite(outFile + os.path.basename(inImage) 
                    + "_KEY_" + str(ImageParse.imgNum) + ".tif", img)

        # Iterate the num of images for naming conventions
        ImageParse.imgNum += 1  


    #--------------------------------------------------------------------------#
    # serializeKey
    # This will perform 2 checks in the input circle to see if it is good data.
    # First we check to make sure no part of the tail goes outside of the image,
    # thus we check to see if our box extends over the boundary of our image. 
    # Second, we check that the circle is not overlaping with another circle. If
    # this is the case then we know that the comets are too close together and
    # it is a bad data value and will be reviewed.
    # @param circ  - The particular circle we are checking to see if good
    # @param circs - The list of circles which will be screened
    # @param img   - The original image for size values
    # @return      - True for bad and false for good
    #--------------------------------------------------------------------------#
    def sanitize(circ, circs, img):
        # Get both the number of circles and the size of the image
        cy, cx = circs.shape
        sy, sx, sz = img.shape

        # First check to see if our circle cut region goes over image edge
        if ((circ[1] - RECT_Y < 0)
                or (circ[1] + RECT_Y > sy)
                or (circ[0] - RECT_X_NEG < 0)
                or (circ[0] + RECT_X_POS > sx)):
            return True

       
        # Check to see if circles overlap, or are within CDIST. 
        for c1 in range(0, cy):
            # This is probably a very slow operation and may req optimization
            if ((not np.array_equal(circ, circs[c1])) 
                    and (((circ[0] - circs[c1][0])**2
                    + (circ[1] - circs[c1][1])**2)
                    < (circ[2] + circs[c1][2] + CDIST)**2)):
                return True 
        return False

    #--------------------------------------------------------------------------#
    # getSplit
    # This extracts the circles from the original image. This leverages and
    # opencv method called Houghs Circles, which seems to be the best wawy to
    # identify circles in an image. It returns a list of the circles detected
    # @param inImage     - The input image name which is to be processed 
    # @return circs, img - The array of circles detected and the img read 
    #--------------------------------------------------------------------------#
    def getSplit(inImage):
        # Open the original image based on the input file name
        img = cv2.imread(inImage)

        # Convert the image to grayscale as that is what houghs requires
        gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        cv2.imshow("B2G", gs)
#        cv2.waitKey(0) 
#        cv2.destroyAllWindows() 
       
        # This uses a Gaussian transform to smooth out singularities
        # The (15,15) region to apply filter at each pix, needs to be refined. 
        gs = cv2.GaussianBlur(gs, BLUR, 0)
#        cv2.imshow("GB", gs)
#        cv2.waitKey(0) 
#        cv2.destroyAllWindows() 
      
        # 4.9 is some ratio of resolution, may need to be refined
        # 50 is the min dist between circles, it is disadvantagous to have this
        # high as we want to detect those circles so we can throw out the bad
        # data 
        # 30 is some screening factor which will through out some bad circs
        circs = cv2.HoughCircles(gs, cv2.HOUGH_GRADIENT, HOUGHS, 50, 100, 30,
                                    minRadius=40, maxRadius=60)
    
        ######## OPTIONAL TO VIEW IMAGES ######## 
        return circs, img
