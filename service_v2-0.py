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
    import run_bn_inference_v2_0 as reasoning_system
    
    # Input should consist of algorithms names, scores, and heat maps
    # NOTE: Hardcoded naming scheme.  A bit hackish.
    # TODO -- Does this need better error correction?  What if score or
    # heatmap is missing?
    log.info('Collecting algorithm scores and heatmaps')
    algs = input['algorithms'].split(',')
    scores   = {a: input[a + '_score'] for a in algs}
    heatmaps = {a: deserialize_image(input[a + '_heatmap'].data) for a in algs}

    log.info('Running probabilistic inference')

    ###############################
    # TODO -- run inference here.
    ###############################
    manips, confidences, heatmaps = reasoning_system.run_inference(scores, heatmaps)

    # STUB -- replace this with something like the commented out line above.
    #manips, confidences, heatmaps = [], [], []

    #
    # Generate output dictionary
    #
    log.info('Collecting and returning inference results')
    output = {}

    # List of all manipulations that we're making predictions for
    output['manipulations'] = ",".join(manips)

    # Set confidence and heatmap for each manipulation
    # (Heatmap value could be "None" if unavailable.)
    for m, c, h in zip(manips, confidences, heatmaps):
        output[m + '_confidence'] = h
        output[m + '_heatmap'] = c

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
    heatmap1 = fileutil.read_binary('../../../../exp_pt1/block02/mask/NC2016_0016.png')
    heatmap2 = fileutil.read_binary('../../../../exp_pt1/dct01/mask/NC2016_0016.png')
    id = uuid4()
    input = {
        'algorithms': 'block01,block02,copymove01,dct01,dct02,dct03_A,dct03_NA,ela01,noise01,noise02',
        'block01_score': 'nan',
        'block01_heatmap': Resource('image', heatmap1, 'image/jpeg'),
        'block02_score': 12.34,
        'block02_heatmap': Resource('image', heatmap1, 'image/jpeg'),
        'dct01_score':  0.0,
        'dct01_heatmap': Resource('image', heatmap2, 'image/jpeg'),
        'dct02_score':  0.0,
        'dct02_heatmap': Resource('image', heatmap1, 'image/jpeg'),
        'dct03_A_score':  0.0,
        'dct03_A_heatmap': Resource('image', heatmap1, 'image/jpeg'),
        'dct03_NA_score':  0.0,
        'dct03_NA_heatmap': Resource('image', heatmap1, 'image/jpeg'),
        'ela01_score':  0.0,
        'ela01_heatmap': Resource('image', heatmap1, 'image/jpeg'),
        'noise01_score':  0.0,
        'noise01_heatmap': Resource('image', heatmap1, 'image/jpeg'),
        'noise02_score':  0.0,
        'noise02_heatmap': Resource('image', heatmap1, 'image/jpeg')
    }
    log = logging.getLogger(__name__)
    process_run(id, input, log)
