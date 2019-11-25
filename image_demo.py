import sys
import argparse
from yolo import YOLO
from PIL import Image

def detect_img(yolo):
    count=1
    #path='C:/Users/Rashid Ali/Desktop/deep_sort_yolov3/elevator_dataset'
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            rimage =yolo.detect_image_2(image)
            rimage.save('./output/'+str(count)+'.PNG')
            count+=1
            rimage.show()
    yolo.close_session()

#def detect_video(yolo):
#video_path = "C:/Users/Rashid Ali/Desktop/deep_sort_yolov3/elevator_dataset/V01.mp4"
#output_path= "C:/Users/Rashid Ali/Desktop/deep_sort_yolov3/elevator_dataset/out_V01.mp4"
#yolo.detect_video(yolo, video_path, output_path)
            
    

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
'''
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO())
    #elif "input" in FLAGS:
    #detect_video(YOLO())
    #else:print("Must specify at least video_input_path.  See usage with --help.")
