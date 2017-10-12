from __future__ import print_function
import sys, pdb
sys.path.insert(0, '.')
import vgg, time
import tensorflow as tf, numpy as np, os
import stylenet
from argparse import ArgumentParser
from vgg import read_img, list_files

vgg_path = 'vgg19.mat'

def build_parser():    
    parser = ArgumentParser(description='Real-time style transfer')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', default='../../train2014s', type=str,
                        help='dataset directory path (according to the paper, use MSCOCO 80k images)')
    parser.add_argument('--style_image', '-s', type=str, required=True,
                        help='style image path')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='batch size (default value is 1)')
    parser.add_argument('--ckpt', '-c', default='ckpt', type=str,
                        help='the global step of checkpoint file desired to restore.')
    parser.add_argument('--lambda_tv', '-l_tv', default=2e2, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--lambda_feat', '-l_feat', default=7.5e0, type=float)
    parser.add_argument('--lambda_style', '-l_style', default=1e2, type=float)
    parser.add_argument('--epoch', '-e', default=2, type=int)
    parser.add_argument('--lr', '-l', default=1e-3, type=float)

    return parser
    

def main():
    parser = build_parser()
    options = parser.parse_args()
        
    if options.gpu > -1:
        device = '/gpu:{}'.format(options.gpu)
    else:
        device = '/cpu:0'

    batchsize = options.batchsize    

    # content targets
    content_targets = [os.path.join(options.dataset, fn) for fn in list_files(options.dataset)]    
    content_targets = content_targets[:-(len(content_targets) % batchsize)] 

    print('total training data size: ', len(content_targets))
    batch_shape = (batchsize,224,224,3)

    # style target
    style_target = read_img(options.style_image)
    style_shape = (1,) + style_target.shape

    with tf.device(device), tf.Session() as sess:

        # style target feature
        # compute gram maxtrix of style target
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        vggstyletarget = vgg.net(vgg_path, vgg.preprocess(style_image))
        style_vgg = vgg.get_style_vgg(vggstyletarget, style_image, np.array([style_target]))        

        # content target feature 
        content_vgg = {}
        inputs = tf.placeholder(tf.float32, shape=batch_shape, name="inputs")
        content_net = vgg.net(vgg_path, vgg.preprocess(inputs))
        content_vgg['relu4_2'] = content_net['relu4_2']

        # feature after transformation 
        outputs = stylenet.net(inputs/255.0)        
        vggoutputs = vgg.net(vgg_path, vgg.preprocess(outputs))

        # compute feature loss
        loss_f = options.lambda_feat * vgg.total_content_loss(vggoutputs, content_vgg, batchsize)

        # compute style loss        
        loss_s = options.lambda_style * vgg.total_style_loss(vggoutputs, style_vgg, batchsize)
        
        # total variation denoising
        loss_tv = options.lambda_tv * vgg.total_variation_regularization(outputs, batchsize, batch_shape)
        
        # total loss
        loss = loss_f + loss_s + loss_tv

        
    with tf.Session() as sess:    
                
        if not os.path.exists(options.ckpt):
            os.makedirs(options.ckpt)
        save_path = os.path.join(options.ckpt,'1.ckpt')

        #training
        train_step = tf.train.AdamOptimizer(options.lr).minimize(loss)
        sess.run(tf.global_variables_initializer())        
    
        total_step = 0
        for epoch in range(options.epoch):
            print('epoch: ', epoch)
            step = 0
            while step * batchsize < len(content_targets):
                time_start = time.time()
                
                batch = np.zeros(batch_shape, dtype=np.float32)
                for i, img in enumerate(content_targets[step * batchsize : (step + 1) * batchsize]):
                   batch[i] = read_img(img).astype(np.float32) # (224,224,3)

                step += 1
                total_step += 1
            
                loss_, _= sess.run([loss, train_step,], feed_dict= {inputs:batch})
                
                time_elapse = time.time() - time_start
                
                should_save = total_step % 2000 == 0                
               
                if total_step % 1 == 0:
                    print('[step {}] elapse time: {} loss: {}'.format(total_step, time_elapse, loss_))

                if should_save:                                        
                    saver = tf.train.Saver()
                    res = saver.save(sess, save_path)
                    print('Save checkpoint')
        


if __name__ == '__main__':
    main()
