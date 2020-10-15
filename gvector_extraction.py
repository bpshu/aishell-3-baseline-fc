import numpy as np 
import os 
import tensorflow as tf 
import glob 
import time 
import random 
import argparse

from functools import partial
from tqdm import tqdm 
from collections import namedtuple 
from typing import Sequence, Tuple, List

## (shiyao) TODO : adjust these imports
from feedback_synthesizer.hparams import hparams
from yyg_g_vector.resnet import ResNet
from feedback_synthesizer.models.embedding.Resnet import resnet_hparams

trainline = namedtuple(
    'trainline', 
    [
        'audio',
        'mel',
        'embed',
        'nsample',
        'nframe',
        'text'
    ]
)

def build_and_restore_resnet(sess : tf.Session, ckpt_path : str) -> Tuple[tf.placeholder, tf.placeholder] : 
    """
    build and restore resnet in a given session, return placehoders for input and output values
    :param sess: tensorflow session in which the graph state is to be restored
    :param ckpt_path: path to saved model chekcpoint to load state from

    :return: a tuple of tf.Placeholder s, `mel_input` and `gv`, which are the input and output placehoders
    for the resnet encoder
    """

    with tf.variable_scope('resnet', reuse = tf.AUTO_REUSE) as resnet_scopt : 
        mel_input = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_input')
        labels    = tf.placeholder(tf.int32, (None), name='labels')

        net       = ResNet(resnet_hparams, mel_input, labels, 'eval')
        net.build_graph()

    ckpt_state = tf.train.get_checkpoint_state(ckpt_path)
    saver      = tf.train.Saver()
    
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    return mel_input, net.gv

def parse_metafile(filename : str) -> List[trainline] : 
    with open(filename) as f : 
        return [
            trainline(*line.strip().split('|'))
            for line in f
        ]

def audio_path_from_metaline(l : trainline) -> str : 
    return os.path.join(
        'audio',
        l.spkid,
        f'{l.sid}.wav'
    )

def _embedding_from_line(sess : tf.Session, nodes : Tuple[tf.placeholder, tf.placeholder], mel_dir : str, l : trainline) -> np.ndarray : 
    
    mel_node, gv_node = nodes

    # load the mel-spectrogram 
    mel = np.load(os.path.join(mel_dir, l.mel))

    gvector = sess.run(
        gv_node,
        {
            mel_node : mel[None]
        }
    )

    return gvector[0]


def _embedding_from_line_direct(sess : tf.Session, nodes : Tuple[tf.placeholder, tf.placeholder],p_mel : str) -> np.ndarray : 
    
    mel_node, gv_node = nodes

    # load the mel-spectrogram
    # mel = np.load(p_mel).T  # when used for generated mel
    mel = np.load(p_mel)    # when used for extracted mel
    # the difference being axis order

    gvector = sess.run(
        gv_node,
        {
            mel_node : mel[None]
        }
    )

    return gvector[0]


def main() -> None : 

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='path to dataset dir, containing train.txt and mel-spectrograms', type=str)
    parser.add_argument('--gvec_ckpt', help='path ot resnet checkpoint', type=str, default='synthesizer_with_encoder/models/embedding/save_models')
    args = parser.parse_args()

    metafile = parse_metafile(os.path.join(args.dataset_dir, 'train.txt'))

    embed_dir = os.path.join(args.dataset_dir, 'embeds')
    mel_dir   = os.path.join(args.dataset_dir, 'mels')
    os.makedirs(embed_dir, exist_ok=True)


    # load network and parameters 
    with tf.Session() as sess : 
        node = build_and_restore_resnet(sess, args.gvec_ckpt)
        embedding_from_line = partial(_embedding_from_line, sess, node, mel_dir)

        for tl in tqdm(metafile) : 
            embedding = embedding_from_line(tl)
            np.save(os.path.join(embed_dir, tl.embed), embedding)

        print(f'n of metafile : {len(metafile)}')
        print(f'e.g. {random.sample(metafile, 5)}')


def main_tsne() : 
    
    gvec_ckpt = 'synthesizer_with_encoder/models/embedding/save_models'
    
    mel_dir = 't_sne_data/trial_4/dep_mels'
    out_dir = 't_sne_data/trial_4/dep_embeds'

    os.makedirs(out_dir, exist_ok=True)
    
    with tf.Session() as sess : 
        node = build_and_restore_resnet(sess, gvec_ckpt)
        embedding_from_line = partial(_embedding_from_line_direct, sess, node)
        
        for mel_path in tqdm(glob.glob(os.path.join(mel_dir, '*.npy'))) : 
            embedding = embedding_from_line(mel_path)
            np.save(os.path.join(out_dir, os.path.basename(mel_path)), embedding)




if __name__ == '__main__' : 
    main()
    # main_tsne()