import cv2
import glob

DATA_PATH = './training/'
OUTPUT_PATH = './sketch_training/'

def img2sktch(image, sketch_image, k_size = 7):
    #Read Image
    img = cv2.imread(image)
    
    # Convert to Grey Image
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img = cv2.bitwise_not(grey_img)
    #invert_img = 255 - grey_img

    # Blur image
    blur_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)

    # Invert Blurred Image
    invblur_img = cv2.bitwise_not(blur_img)
    #invblur_img = 255 - blur_img

    # Sketch Image
    sktch_img = cv2.divide(grey_img, invblur_img, scale = 256.0)

    # Save Sketch 
    cv2.imwrite(sketch_image, sktch_img)
    
#Function call
print('Start Converting...')
count = 0
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spyder', 'squirrel']
# classes = ['butterfly']
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(DATA_PATH + label +'/*'): # sheep
        sketch = image.replace("training", "sketch_training")
        img2sktch(image, sketch)
        count += 1
        if count % 1000 == 1:
            print(str(round(float(count / 25139 * 100), 2)) + '%')

# def img2sktch(image, sketch_image, k_size = 7):
    # #Read Image
    # img = torchvision.io.read_image(image)

    # # Convert to Grey Image
    # grey_img = torchvision.transforms.Grayscale(num_output_channels = 1)(img)

    # # Invert Image
    # invert_img = torch.bitwise_not(grey_img, dtype=torch.int8)
    # #invert_img = 255 - grey_img

    # # Blur image
    # blur_img = torchvision.transforms.GaussianBlur(k_size, sigma = 0)(invert_img)

    # # Invert Blurred Image
    # invblur_img = torch.bitwise_not(blur_img, dtype=torch.int8)
    # #invblur_img = 255 - blur_img

    # # Sketch Image
    # sktch_img = torch.div(blur_img, invblur_img) * 256

    # # Save Sketch 
    # torchvision.utils.save_image(sktch_img, sketch_image)