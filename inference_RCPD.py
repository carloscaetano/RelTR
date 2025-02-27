# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
import json
import logging
import os
from vg_classes import CLASSES, REL_CLASSES
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")
    parser.add_argument('--folder_path', type=str, help="Folder path contatining RCPD images")
    parser.add_argument('--output_folder', type=str, help="Folder path to save the json and feature maps")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing: cuda or cpu')
    parser.add_argument('--resume', default='ckpt/checkpoint0149_oi.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def run_on_folder(folder_path, output_folder, model):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_img_triplets = {}
    for img_path in os.listdir(folder_path):
        try:
            print("Opening {} ".format(img_path), flush=True, end="")
            img_triplets = []
            im = Image.open(os.path.join(folder_path, img_path))

            # mean-std normalize the input image (batch-size: 1)
            img = transform(im).unsqueeze(0)

            # propagate through the model
            outputs = model(img)
            print("... propagating through the model ", flush=True, end="")

            # keep only predictions with 0.+ confidence
            probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
            probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
            probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
            keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                                    probas_obj.max(-1).values > 0.3))

            # convert boxes from [0; 1] to image scales
            sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
            obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

            if len(sub_bboxes_scaled) == 0:
                error_msg = "Error processing: {}. ZERO detections on image!".format(os.path.join(folder_path, img_path))
                logging.exception(error_msg)
                print(error_msg)
                continue

            topk = 10
            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
            keep_queries = keep_queries[indices]

            # use lists to store the outputs via up-values
            conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

            hooks = [
                model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)
                ),
                model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_sub.append(output[1])
                ),
                model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_obj.append(output[1])
                )
            ]
            with torch.no_grad():
                # propagate through the model
                print("..", flush=True, end="")
                outputs = model(img)
                print(". ", flush=True, end="")

                for hook in hooks:
                    hook.remove()

                # don't need the list anymore
                conv_features = conv_features[0]
                dec_attn_weights_sub = dec_attn_weights_sub[0]
                dec_attn_weights_obj = dec_attn_weights_obj[0]

                # get the feature map shape
                h, w = conv_features['0'].tensors.shape[-2:]
                im_w, im_h = im.size
                print("generating output ...", flush=True, end="")
                fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
                for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                        zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                    
                    if len(sub_bboxes_scaled) == 1:
                        ax_i = axs.T
                    
                    ax = ax_i[0]
                    ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
                    ax.axis('off')
                    ax.set_title(f'query id: {idx.item()}')
                    ax = ax_i[1]
                    ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
                    ax.axis('off')
                    ax = ax_i[2]
                    blank_im = Image.new('RGB', (im_w, im_h))
                    ax.imshow(blank_im)
                    ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                            fill=False, color='blue', linewidth=2.5))
                    ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                            fill=False, color='orange', linewidth=2.5))

                    ax.axis('off')
                    ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)
                    
                    triplet_i = {
                        "query_id": int(idx),
                        "sub": CLASSES[probas_sub[idx].argmax()],
                        "pred": REL_CLASSES[probas[idx].argmax()],
                        "obj": CLASSES[probas_obj[idx].argmax()]
                        }
                    
                    img_triplets.append(triplet_i)
            
                fig.tight_layout()
                output_img_filepath = os.path.join(output_folder, img_path)
                plt.savefig(output_img_filepath)
                plt.cla()
                plt.close(fig)
                print("done!", flush=True)

            all_img_triplets[img_path] = img_triplets
        except:
            error_msg = "Error processing: {}".format(os.path.join(folder_path, img_path))
            logging.exception(error_msg)
            print(error_msg)
    
    triplet_json_filepath = os.path.join(output_folder, 'SG_triplets.json')
    with open(triplet_json_filepath, 'w') as json_file:
        json.dump(all_img_triplets, json_file)


def main(args):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    logging.basicConfig(filemode='w', level=logging.ERROR, filename=os.path.join(args.output_folder, 'log_file.txt'))

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume, map_location=torch.device(args.device))
    model.load_state_dict(ckpt['model'])
    model.eval()
    run_on_folder(args.folder_path, args.output_folder, model)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
