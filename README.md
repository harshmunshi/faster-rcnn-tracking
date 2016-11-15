# faster-rcnn-tracking

A simple tracking integration which can take the bounding box information using faster rcnn and processes it using sort algorithm as developed by https://github.com/abewley/sort

# Dependencies

The code requires all the dependencies as requires in py-faster-rcnn and sort algorithm. To get started please follow the following links:
   
    1. https://github.com/rbgirshick/py-faster-rcnn
    2. https://github.com/abewley/sort

and install all the dependencies as mentioned there.

Furthermore the python version used in this algorithm is 2.7.12, but it should work fine with 2.7.x

# Using the code

Assuming that you have already installed the dependencies, navigate to the tools folder in the faster RCNN folder

    $ cd FRCN_ROOT/tools/
    $ git clone https://github.com/harshmunshi/faster-rcnn-tracking.git
    $ cd ..
    $ mv ./faster-rcnn-tracking/sort.py ./
    $ mv ./faster-rcnn-tracking/sort_demo.py ./
    $ rm -rf faster-rcnn-tracking
    
Now both the files should be under /FRCN_ROOT/tools. To run the code, 

    $ cd FRCN_ROOT/tools
    $ python sort.py /path/to/<video_source>

# Note

This algorithm is still under development. Any contribution is highly appreciated.
