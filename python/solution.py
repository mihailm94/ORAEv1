#!/usr/bin/env python3

from typing import (
    List,
    Tuple,
    Union
)

from utils.image import (
    ImageType,
    PackedImage,
    StrideImage,
)

from utils.function_tracer import FunctionTracer

################################################################################
# USER IMPORTS
################################################################################
from utils.pixel import Pixel
from collections import deque


def getEyeBorders(imgW: int, imagePixels: List[Pixel]) -> List[Tuple[int, int, int, int]]:
    """ Finds and returns the coordinates of all eyes in an image.
    In a sense, the coordinates are the "bounding box" of an eye.
    It gives the endmost points (northWest, northEast, southWest, southEast)
    of the eye in terms of pixel indices within an image.
    """

    # First detect top and bottom lines because it is easier to do matrix
    # indexing that way.
    # All eye patterns contain 3 dashes "---" on the top and the bottom.
    # So: 
    # 1. Search for 3 contiguous pixels whose red is >= 200.
    # 2. Check if neighbors (rows below) also have 3 contiguous pixels
    # whose red is >= 200.
    # 3. Then check if left and right edges exist.

    # We define the red eye as having coordinates in the image
    # We are interested in the corner coordinates (NW, NE, SW, SE),
    # as using them will help us delimit the "red eye window".
    # Plus, they indent pretty nicely when written underneath each other!
    #
    # northWest
    # northEast
    # southWest
    # southEast
    # 
    #            North
    #
    #        NW ______ NE
    #           |    |
    #  West     |    |      East
    #           |    |
    #        SW ______ SE
    #
    #            South

    # List of coordinates of all eyes in the image
    # Eye coordinates are: (northWest, northEast, southWest, southEast)
    coordinates: List[Tuple[int, int, int, int]] = []
    # Use deque() because of O(1) pop'n'push
    # Used to iterate over pixels.
    pixels = deque(maxlen=3)
    
    for idp, pixel in enumerate(imagePixels):
        pixels.append(pixel)
        # Red eye condition (red >= 200).
        # These are the top 3 pixels that define an eye.
        # They trigger detection of the rest of the pixels.
        # Basically, a very rudimentary edge detection algorithm.
        if(pixels[0].red >= 200 and pixels[1].red >= 200 and pixels[2].red >= 200):
            # Check if neighbors downstairs got super red pixels too.
            # The column for the pixels downstairs is the same.
            # The row is 4x the width of the image, because that is the eye
            # pattern structure (eye_pattern.py).
            # Pixels are in a list, so just take the pixel index and add 4x
            # the width of the image.
            #
            #     __________ 
            #     |        | 
            #     |     x--| 
            #     |--------|     x: Pixel
            #     |--------|     -: width "measure"
            #     |--------|
            #     |-----x  |
            #     |        | 
            #     __________
            # 

            # First do a sanity check: do not exceed list range!
            # @see logic in the next comment for why (idp - 0 + 4*imgW) makes
            # sense.
            if((idp - 0 + 4*imgW) > len(imagePixels)):
                # No need to check further, skip whole rest of image!
                # Saves some time too, since no further pixel checks are
                # needed.
                break

            # 3 bottom pixels.
            # idp is currently pointing at the last of the 3 pixels (the dashes
            # in the eye pattern).
            # To check the row of dashes downstairs, go back 2 pixels to the 
            # first of the bottom row pixels.
            # _____________________________________________
            # |                    idp                    |
            # |                    |                      | 
            # |                * * X                      |
            # |                                           | 
            # |                 eye                       |    
            # |                                           |
            # |                Y * *                      |
            # |                ^                          |
            # |                |                          |
            # |                idp - 2 + 4*imgW           |
            # _____________________________________________
            pxlB_1 = imagePixels[idp - 2 + 4*imgW].red
            pxlB_2 = imagePixels[idp - 1 + 4*imgW].red
            pxlB_3 = imagePixels[idp - 0 + 4*imgW].red
            # Check if bottom pixels are part of an eye
            if(pxlB_1 >= 200 and pxlB_2 >= 200 and pxlB_3 >= 200):
                eyeCoords = getEyeCoordinates(pixelId=idp,
                                                imgW=imgW,
                                                imagePixels=imagePixels)
                # Will be None if left or right borders are missing.
                if eyeCoords is not None:
                    coordinates.append(eyeCoords)
                else:
                    # Left or Right edges of eye are nonexistent.
                    # Skip pixel.
                    continue

    return coordinates


def getEyeCoordinates(pixelId: int, imgW: int, imagePixels: List[Pixel]) -> Union[None, Tuple[int, int, int, int]]:
    """ Find whether the left and right pixels of an eye have red values above
    200.
    Caller function is @see findTopBottom()
    Assumes that the top and bottom pixels of an eye are already "found"

    """
    # The top and bottom pixels may be part of an eye.
    # Check if side pixels also have red values >= 200
    # 6 side pixels (3 left, 3 right).
    #
    # Note: We can safely increment the pixelId counter because of the sanity
    # check in the getEyeBorders() function.

    # Right
    pxlR_1 = imagePixels[pixelId + 1 + 1*imgW].red
    pxlR_2 = imagePixels[pixelId + 1 + 2*imgW].red
    pxlR_3 = imagePixels[pixelId + 1 + 3*imgW].red
    # Left
    pxlL_1 = imagePixels[pixelId - 3 + 1*imgW].red
    pxlL_2 = imagePixels[pixelId - 3 + 2*imgW].red
    pxlL_3 = imagePixels[pixelId - 3 + 3*imgW].red

    # Red eye condition
    if(pxlR_1 >= 200 and pxlR_2 >= 200 and pxlR_3 >= 200 and\
        pxlL_1 >= 200 and pxlL_2 >= 200 and pxlL_3 >= 200):

        # We assume with high degree of certainty that there is
        # an eye.
        # Get the NW, NE, SW, SE coordinates of an eye
        #
        # _____________________________________________
        # |                    idp                    |
        # |                    |                      | 
        # |  idp - 3 >   * * * * * < idp + 1          |
        # |              *       * < idp + 1 + 1*w    |    idp  = pixelId
        # |              *  eye  * < idp + 1 + 2*w    |    w    = imgW
        # |              *       * < idp + 1 + 3*w    |
        # |              * * * * * < idp + 1 + 4*w    |
        # |              ^                            |
        # |              |                            |
        # |              idp - 3 + 4*w                |
        # _____________________________________________
        northWest = pixelId - 3
        northEast = pixelId + 1
        southWest = pixelId - 3 + 4*imgW
        southEast = pixelId + 1 + 4*imgW

        # Coordinates of eye returned as a tuple
        return (northWest, northEast, southWest, southEast)
    
    # No left or right edges of eye
    return None


def getEyeInnerPixels(imgW: int, border: Tuple[int, int, int, int], imagePixels: List[Pixel]) -> List[int]:
    """ Scan the "inner" part of an eye for any pixels that have a red
    value higher than 200.
    The eye border is passed as an argument. 
    This function is hard coded to use the eye patterns as defined in
    eye_pattern.py

    """
    innerPixels: List[int] = []

    # First row of eye contains only the eye boundary, no inner pattern.
    # Skip first row and start from the "white part" of the eye on the
    # second row.
    eyeIndex = border[0] + imgW + 1 
    # _____________________________________________
    # |                                           |
    # |                                           | 
    # |              * * * * *                    |
    # |              * X     *                    | eyeIndex has value of X
    # |              *  eye  *                    |    
    # |              *       *                    |
    # |              * * * * *                    |
    # |                                           |
    # _____________________________________________

    # EYE_PATTERN_1
    if(imagePixels[eyeIndex].red < 200 and\
        imagePixels[eyeIndex+1].red < 200 and imagePixels[eyeIndex+2].red < 200):
        innerPixels.append(eyeIndex + imgW)
        innerPixels.append(eyeIndex + imgW + 1)
        innerPixels.append(eyeIndex + imgW + 2)
        # _____________________________________________
        # |                                           |
        # |                                           | 
        # |              * * * * *                    |
        # |              *       *                    | 
        # |              * X X X *                    |    
        # |              *       *                    |
        # |              * * * * *                    |
        # |                                           |
        # _____________________________________________

        return innerPixels
    
    # EYE_PATTERN_2 & EYE_PATTERN_3
    if(imagePixels[eyeIndex].red < 200 and imagePixels[eyeIndex+1].red >= 200):
        # EYE_PATTERN_2
        innerPixels.append(eyeIndex + 1)
        innerPixels.append(eyeIndex + imgW + 1)
        innerPixels.append(eyeIndex + imgW*2 + 1)
        # _____________________________________________
        # |                                           |
        # |                                           | 
        # |              * * * * *                    |
        # |              *   X   *                    | 
        # |              *   X   *                    |    
        # |              *   X   *                    |
        # |              * * * * *                    |
        # |                                           |
        # _____________________________________________
        # EYE_PATTERN_3
        if(imagePixels[eyeIndex+imgW].red >= 200):
            innerPixels.append(eyeIndex+imgW)
            innerPixels.append(eyeIndex+imgW+2)
            # _____________________________________________
            # |                                           |
            # |                                           | 
            # |              * * * * *                    |
            # |              *   X   *                    | 
            # |              *       *                    |    
            # |              *   X   *                    |
            # |              * * * * *                    |
            # |                                           |
            # _____________________________________________

        return innerPixels

    # EYE_PATTERN_4 is unique in that the first pixel is already indicative of
    # the pattern.
    if(imagePixels[eyeIndex].red >= 200):
        # Eye pattern 4
        innerPixels.append(eyeIndex)
        innerPixels.append(eyeIndex + 2)
        innerPixels.append(eyeIndex + imgW + 1)
        innerPixels.append(eyeIndex + imgW*2)
        innerPixels.append(eyeIndex + imgW*2 + 2)
        # _____________________________________________
        # |                                           |
        # |                                           | 
        # |              * * * * *                    |
        # |              * X   X *                    | 
        # |              *   X   *                    |    
        # |              * X   X *                    |
        # |              * * * * *                    |
        # |                                           |
        # _____________________________________________

        return innerPixels
    
    return None


def getAllEyePixels(imgW: int, border: Tuple[int, int, int, int], eyeInner: List[int]) -> List[int]:
    """ Get the pixels of an eye as indices of the pixel list.
    Can be then used to iterate over the pixel values and decrement the red
    value.
    
    """
    allEyePixels: List[int] = []

    # Inner pattern
    allEyePixels.extend(eyeInner)

    # Top border, with the range being inclusive ( [] not [) )
    allEyePixels.extend(range(border[0], border[1]+1))

    # Side borders
    # Optimization: no need for a loop. Unrolling
    # Left borders
    allEyePixels.append(border[0]+imgW)
    allEyePixels.append(border[0]+imgW*2)
    allEyePixels.append(border[0]+imgW*3)
    # Right borders
    allEyePixels.append(border[1]+imgW)
    allEyePixels.append(border[1]+imgW*2)
    allEyePixels.append(border[1]+imgW*3)

    # Bottom border, with the range being inclusive ( [] not [) )
    allEyePixels.extend(range(border[2], border[3]+1))

    return allEyePixels



def compute_solution(images: List[Union[PackedImage, StrideImage]]):
    ft = FunctionTracer("compute_solution", "seconds")
    
    for img in images:
        # Width needed for finding neighboring pixels (vertically)
        w = img.resolution.width
        # All pixels which constitute an eye
        eyePixels: List[int] = []

        # [(northWest, northEast, southWest, southEast), (),  ...]
        eyeBorders = getEyeBorders(imgW=w, imagePixels=img.pixels)

        for border in eyeBorders:
            # Get the pixels which are within the eye boundaries
            eyeInner = getEyeInnerPixels(imgW=w, border=border, imagePixels=img.pixels)
            # All eye pixels, inner, or border pixels, returned as indices
            # for idexing the img.pixels[] list.
            eyePixels.extend(getAllEyePixels(imgW=w, border=border, eyeInner=eyeInner))

        for pixel in eyePixels:
            # Reduce pixel values
            img.pixels[pixel].red -= 150

    del ft

# EOF

