import io

import numpy as np
from scipy import ndimage
from scipy.misc import imread, imsave

from medifor import fileutil, processing
from medifor.resources import Resource

name = 'Reasoning Service'
max_run_count = 16

def process_runs(run_queue):
    return processing.process_runs_sequential(run_queue, process_run)

def process_run(id, input, log):
    import generate_Medifor_bn_model_10 as reasoning_system
    #The line below training the model can be omitted if the base bayesian network file is already in the working directory.
    #reasoning_system.train_model()
    
    # Input should consist of algorithms names, scores, and heat maps
    # NOTE: Hardcoded naming scheme.  A bit hackish.
    # TODO -- Does this need better error correction?  What if score or
    # heatmap is missing?
    log.info('Collecting algorithm scores and heatmaps')
    #algs = input['algorithms'].split(',')
    #scores   = {a: input[a + '_score'] for a in algs}
    #heatmaps = {a: deserialize_image(input[a + '_heatmap'].data) for a in algs}

    log.info('Running probabilistic inference')

    ###############################
    # TODO -- run inference here.
    ###############################
    results = reasoning_system.run_inference(input)
    #manips, confidences, heatmaps = reasoning_system.run_inference(scores, heatmaps)

    # STUB -- replace this with something like the commented out line above.
    #manips, confidences, heatmaps = [], [], []

    #
    # Generate output dictionary
    #
    log.info('Collecting and returning inference results')
    output = results

    # List of all manipulations that we're making predictions for
    #output['manipulations'] = ",".join(manips)

    # Set confidence and heatmap for each manipulation
    # (Heatmap value could be "None" if unavailable.)
    #for m, c, h in zip(manips, confidences, heatmaps):
    #    output[m + '_confidence'] = h
    #    output[m + '_heatmap'] = c

    return output


def deserialize_image(data, flatten=False):
    with io.BytesIO(data) as stream:
        return imread(stream, flatten=flatten)

def serialize_image(image, format='jpeg'):
    with io.BytesIO() as stream:
        imsave(stream, image, format=format)
        return stream.getvalue()


if __name__ == '__main__':
    import logging
    import sys
    from uuid import uuid4

    # TODO -- Create better test input...
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')
    #heatmap1 = fileutil.read_binary('../../../../exp_pt1/block02/mask/NC2016_0016.png')dd
    heatmap1 = imread('NC2016_0016_block02.png')
    #heatmap2 = fileutil.read_binary('../../../../exp_pt1/dct01/mask/NC2016_0016.png')
    heatmap2 = imread('NC2016_0016_dct01.png')
    query_image = imread('NC2016_0016.jpg')
    id = uuid4()
    input = {
        # The input format requires the query_image, a list of relevant algorithms (must be complete), and any evidence the user has regarding the algorithms (i.e. scores and/or heatmaps).
        'image': Resource('image', query_image, 'image/jpeg'),
        'algorithms': ['block01', 'block02', 'copymove01', 'dct01', 'dct02', 'dct03_A', 'dct03_NA', 'ela01', 'noise01', 'noise02'],
        'block02': {'heatmap': Resource('image', heatmap1, 'image/png')},
        'dct01': {'score': '12.04', 'heatmap': Resource('image', heatmap2, 'image/png')}
    }
    log = logging.getLogger(__name__)
    output = process_run(id, input, log)
    #dictionary of results is output key are name of manipulation, value is another dictionary with another heatmap and confidence level
    #[manipulation name]['heatmap'].data - will give actual image object - that will need to pass into a function for actual display image 
    output['removal'].heatmap.data.show()
    #so the output[removal][heamap].data is the same as image.jpg
    print(output)
