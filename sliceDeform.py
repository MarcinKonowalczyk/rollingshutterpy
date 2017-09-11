''' Functions for deformation of image slice
'''

from PIL import Image

import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize

# Import hilbert curve
import sys
#sys.path.append('./hilbert_curve')
sys.path.insert(1,'./hilbert_curve')
from hilbert import distance_from_coordinates as c2d
#from hilbert import coordinates_from_distance as d2c

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def clamp_nodes(nodes):
    ''' Make sure nodes don't move too far
    '''
    nodes[0] = 0
    nodes[-1] = 0

    # Clamp to reasonable offsets
    for x, node in enumerate(nodes):
        nodes[x] = clamp(node,-0.9,0.9)

    # Clamp to limits due to neightbours
    for x, node in enumerate(nodes):
        if x == 0 or x == len(nodes)-1:
            continue
        node = clamp(node,-1,1)
        nodes[x] = clamp(node,nodes[x-1]-1,nodes[x+1]+1)
    return nodes

def deform(image, nodes):
    ''' Deform image acording to `nodes`
    '''
    output_slice = image.copy() # Take copy of the input slice
    w, h = image.size

    nodes = clamp_nodes(nodes) # Make sure nodes are not ridiculous
    dn = w/(len(nodes)-1) # Width between the nodes in pixels
    nodes = [i*dn for i in nodes] # Scale by width
    x = np.linspace(0,w,len(nodes))

    f = CubicSpline(x, nodes, bc_type=('clamped','clamped'))
    # f = interp1d(x, nodes, kind='linear')

    # plt.plot(x,x+nodes)
    # plt.show()
    pixels = list(image.getdata())

    for ri in range(0,h):
        # Figure out scaled interpolant
        def interpolant(index):
            if h == 1:
                blend = 1
            else:
                blend = (1-ri/(h-1))
            return index + blend * f(np.array([index]))[0]
        
        row = pixels[0+ri*w:w+ri*w]
        for ci, pixel in enumerate(row):
            irp = interpolant(ci)
            lower = np.floor(irp)
            blend = irp-lower
            lower = np.uint16(clamp(lower,0,w-1))

            out = [0,0,0]
            for ii, channel in enumerate(row[lower]): # add left pixel
                out[ii] = (1-blend)*channel
            if lower != w-1:
                for ii, channel in enumerate(row[lower+1]): # add right pixel
                    out[ii] += blend*channel

            output_slice.putpixel((ci,ri),tuple(np.uint8(out)))

    return output_slice

def image2harray(image):
    ''' Converts image to hilbert color space numpy array
    '''
    w,h = image.size
    max_hilbert = 2**(8*3)-1 # Maximum value of RGB color in hilbert space
    image_hilbert = np.array([c2d(list(i),8,3)/max_hilbert for i in image.getdata()])
    return image_hilbert.reshape(w,h)

def image2hcolor(image):
    ''' Converts image to hilbert color space image
    '''
    image_hilbert = np.uint8(image2harray(image)*255)
    return Image.fromarray(image_hilbert,'L')

def image2garray(image):
    ''' Converts image to grayscale numpy array
    '''
    w, h = image.size
    image_gray = np.fromstring(image.convert('L').tobytes(), dtype=np.uint8)/255
    return image_gray.reshape(w,h)

def image2gcolor(image):
    ''' Converts image to grayspace image
    '''
    image_gray = np.uint8(image2garray(image)*255)
    return Image.fromarray(image_gray,'L')

def fit(image_1, image_2, mode='gray', n=0, tol=1, maxiter=10):
    ''' Fits the top edge of image_2 (source) to the bottom edge of image_1 (target)
    If 'n' == 0, the number of nodes in the deformation is picked automatically 
    '''
    w, h = image_1.size
    target = image_1.crop((0,h-1,w,h))
    source = image_2.crop((0,0,w,1))

    # Choose mode of quantifying the image
    if mode == 'hilbert':
        transform_function = image2harray
    elif mode == 'gray':
        transform_function = image2garray
    else:
        pass

    target = transform_function(target)

    def fitfun(nodes):
        ''' Mean square difference between deformed and target image
        '''
        nodes = [0] + nodes + [0] # Add edge points to nodes
        deformed = deform(source,nodes)
        deformed = transform_function(deformed)
        diff = (deformed - target)**2
        print(nodes)
        return diff.sum()

    # Nodes to deform the image with
    if n == 0: n = np.ceil(np.sqrt(w))
    nodes0 = np.zeros(np.uint16(n))

    # Fit the image edge
    options = {'gtol': tol, 'disp': True, 'eps': 0.05, 'maxiter': maxiter}
    res = minimize(fitfun, nodes0, method='CG', options=options)

    nodes_fit = [0] + res.x + [0]

    return deform(image_2,nodes_fit)

def main():
    lena = Image.open("./lena.png")
    w, h = lena.size

    slice1 = lena.crop((0,0,w,h//2))
    slice2 = lena.crop((0,h//2,w,h))
    
    slice1 = fit(slice2, slice1)

    im = Image.new('RGB',(w,h))
    im.paste(slice2,(0,0,w,h//2))
    im.paste(slice1,(0,h//2,w,h))
    im.show()

if __name__ == '__main__':
    main()