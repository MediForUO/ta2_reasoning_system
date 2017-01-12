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
    import MediFor_bn_model_generator_13 as reasoning_system
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
    results = reasoning_system.run_inference(input, log)
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

    # TODO -- Create better test input~.
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')
    #heatmap1 = fileutil.read_binary('~/../../../exp_pt1/block02/mask/NC2016_0016.png')dd
    #abb01_heatmap = imread('~/exp_pt3/block01/mask/NC2016_0016.png')
    block01_heatmap = imread('/home/robert/exp_pt2/block01/mask/NC2016_0016.png')
    block02_heatmap = imread('/home/robert/exp_pt1/block02/mask/NC2016_0016.png')
    cfa01_heatmap = imread('/home/robert/exp_pt3/cfa01/mask/NC2016_0016.png')
    cfa02_heatmap = imread('/home/robert/exp_pt3/cfa02/mask/NC2016_0016.png')
    copymove01_heatmap = imread('/home/robert/exp_pt2/copymove01/mask/NC2016_0016.png')
    dct01_heatmap = imread('/home/robert/exp_pt1/dct01/mask/NC2016_0016.png')
    dct02_heatmap = imread('/home/robert/exp_pt1/dct02/mask/NC2016_0016.png')
    dct03_A_heatmap = imread('/home/robert/exp_pt1/dct03_A/mask/NC2016_0016.png')
    dct03_NA_heatmap = imread('/home/robert/exp_pt1/dct03_NA/mask/NC2016_0016.png')
    ela01_heatmap = imread('/home/robert/exp_pt1/ela01/mask/NC2016_0016.png')
    noise01_heatmap = imread('/home/robert/exp_pt1/noise01/mask/NC2016_0016.png')
    noise02_heatmap = imread('/home/robert/exp_pt1/noise02/mask/NC2016_0016.png')
    #heatmap2 = fileutil.read_binary('../../../../exp_pt1/dct01/mask/NC2016_0016.png')
    query_image = imread('/home/robert/NC2016_Test0613/probe/NC2016_0016.jpg')
    id = uuid4()
    input = {
        # The input format requires the query_image, a list of relevant algorithms (must be complete), and any evidence the user has regarding the algorithms (i.e. scores and/or heatmaps).
        'image': Resource('image', query_image, 'image/jpeg'),
        'algorithms': ['block01', 'block02', 'cfa01', 'cfa02', 'copymove01', 'dct01', 'dct02', 'dct03_A', 'dct03_NA', 'ela01', 'noise01', 'noise02'],
        'block01': {'heatmap': Resource('image', block01_heatmap, 'image/png')},
        'block02': {'heatmap': Resource('image', block02_heatmap, 'image/png')},
        'cfa01': {'heatmap': Resource('image', cfa01_heatmap, 'image/png')},
        'cfa02': {'heatmap': Resource('image', cfa02_heatmap, 'image/png')},
        'copymove01': {'heatmap': Resource('image', copymove01_heatmap, 'image/png')},
        'dct01': {'heatmap': Resource('image', dct01_heatmap, 'image/png')},
        'dct02': {'heatmap': Resource('image', dct02_heatmap, 'image/png')},
        'dct03_A': {'heatmap': Resource('image', dct03_A_heatmap, 'image/png')},
        'dct03_NA': {'heatmap': Resource('image', dct03_NA_heatmap, 'image/png')},
        'ela01': {'heatmap': Resource('image', ela01_heatmap, 'image/png')},
        'noise01': {'heatmap': Resource('image', noise01_heatmap, 'image/png')},
        'noise02': {'heatmap': Resource('image', noise02_heatmap, 'image/png')}
    }
    input['block01']['score'] = 3133.104492
    input['block02']['score'] = 39.16
    input['cfa01']['score'] = 0.560267
    input['cfa02']['score'] = 0.413683
    input['copymove01']['score'] = 73996
    input['dct01']['score'] = 0.824778
    input['dct02']['score'] = 0.5
    input['dct03_A']['score'] = 1.267323
    input['dct03_NA']['score'] = 0.459998
    input['ela01']['score'] = 0
    input['noise01']['score'] = 2.007974
    input['noise02']['score'] = 130.243256
    log = logging.getLogger(__name__)
    output, log = process_run(id, input, log)
    #dictionary of results is output key are name of manipulation, value is another dictionary with another heatmap and confidence level
    #[manipulation name]['heatmap'].data - will give actual image object - that will need to pass into a function for actual display image 
    print(output)
    for key in output.keys():
        print(str(key) + ' ' + str(output[key].confidence))
        if(output[key].heatmap):
            output[key].heatmap.data.save(key + '_hm_output.png')
    #so the output[removal][heamap].data is the same as image.jpg
