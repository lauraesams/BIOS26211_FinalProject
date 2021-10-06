import numpy

def saccade_change(mu, sigma, nonzero_saccades=True):
    ##Generate microsaccade jump
    #Calculate distance
    r = numpy.random.normal(loc=mu,scale=sigma)

    #Calculate direction
    theta=numpy.random.uniform()*2*numpy.pi
        
    #Calculate cordinate change | pol2cart
    x = int(numpy.round(r * numpy.cos(theta)))
    y = int(numpy.round(r * numpy.sin(theta)))

    #Check for non zero saccade
    if nonzero_saccades:
        while x==0 and y==0:
            #Do another saccade
            x,y = saccade_change(mu, sigma, nonzero_saccades)
    return x,y


def saccade_new(imgs_nbr, img_width, img_height, patch_width, patch_height):
    ##Generates a new saccade position
    img = int(numpy.floor(numpy.random.uniform(0,imgs_nbr)))
    x = int(numpy.floor(numpy.random.uniform(0,img_width-patch_width)))
    y = int(numpy.floor(numpy.random.uniform(0,img_height-patch_height)))

    return img,x,y

        
def saccade_update(curr_x, curr_y, curr_img, saccade_nbr, saccades_perimg, imgs_nbr, img_width, img_height, patch_width, patch_height, mu=0, sigma=2):
    ##Updates the current possition to a new one
    #Image changes every saccades_perimg
    if saccade_nbr % saccades_perimg == 0:
        img,x,y = saccade_new(imgs_nbr, img_width, img_height, patch_width, patch_height)
    else:
        #Microsaccade on the same image
        img=curr_img
        change_x,change_y = saccade_change(mu, sigma, nonzero_saccades=True)
        x = curr_x + change_x
        y = curr_y + change_y
        #If microsaccade goes outside make a new saccade
        if x<0 or y<0 or x>(img_width-patch_width) or y>(img_height-patch_height):
            img,x,y = saccade_new(imgs_nbr, img_width, img_height, patch_width, patch_height)

    return img,x,y
