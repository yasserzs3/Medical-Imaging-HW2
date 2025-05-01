import os
import tensorflow as tf
import numpy as np
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json

def create_model_config(num_classes, batch_size):
    pipeline_config = {
        'model': {
            'ssd': {
                'num_classes': num_classes,
                'image_resizer': {
                    'fixed_shape_resizer': {
                        'height': 320,
                        'width': 320
                    }
                },
                'feature_extractor': {
                    'type': 'ssd_mobilenet_v2_fpn_keras',
                    'depth_multiplier': 1.0,
                    'min_depth': 16,
                    'conv_hyperparams': {
                        'regularizer': {
                            'l2_regularizer': {
                                'weight': 0.00004
                            }
                        },
                        'initializer': {
                            'random_normal_initializer': {
                                'stddev': 0.01,
                                'mean': 0.0
                            }
                        },
                        'activation': 'relu6',
                        'batch_norm': {
                            'decay': 0.997,
                            'epsilon': 0.001,
                            'scale': True
                        }
                    }
                },
                'box_coder': {
                    'faster_rcnn_box_coder': {
                        'y_scale': 10.0,
                        'x_scale': 10.0,
                        'height_scale': 5.0,
                        'width_scale': 5.0
                    }
                },
                'matcher': {
                    'argmax_matcher': {
                        'matched_threshold': 0.5,
                        'unmatched_threshold': 0.5,
                        'ignore_thresholds': False,
                        'negatives_lower_than_unmatched': True,
                        'force_match_for_each_row': True
                    }
                },
                'similarity_calculator': {
                    'iou_similarity': {}
                },
                'box_predictor': {
                    'weight_shared_convolutional_box_predictor': {
                        'conv_hyperparams': {
                            'regularizer': {
                                'l2_regularizer': {
                                    'weight': 0.00004
                                }
                            },
                            'initializer': {
                                'random_normal_initializer': {
                                    'stddev': 0.01,
                                    'mean': 0.0
                                }
                            },
                            'activation': 'relu6',
                            'batch_norm': {
                                'decay': 0.997,
                                'epsilon': 0.001,
                                'scale': True
                            }
                        },
                        'depth': 256,
                        'num_layers_before_predictor': 4,
                        'kernel_size': 3
                    }
                },
                'anchor_generator': {
                    'ssd_anchor_generator': {
                        'num_layers': 6,
                        'min_scale': 0.2,
                        'max_scale': 0.95,
                        'aspect_ratios': [1.0, 2.0, 0.5, 3.0, 0.3333]
                    }
                },
                'post_processing': {
                    'batch_non_max_suppression': {
                        'score_threshold': 0.3,
                        'iou_threshold': 0.6,
                        'max_detections_per_class': 100,
                        'max_total_detections': 100
                    },
                    'score_converter': 'SIGMOID'
                },
                'normalize_loss_by_num_matches': True,
                'loss': {
                    'localization_loss': {
                        'weighted_smooth_l1': {}
                    },
                    'classification_loss': {
                        'weighted_sigmoid_focal': {
                            'alpha': 0.25,
                            'gamma': 2.0
                        }
                    }
                },
                'focal_loss_gamma': 2.0,
                'focal_loss_alpha': 0.25
            }
        },
        'train_config': {
            'batch_size': batch_size,
            'data_augmentation_options': [
                {
                    'random_horizontal_flip': {}
                },
                {
                    'random_crop_image': {
                        'min_object_covered': 0.0,
                        'min_aspect_ratio': 0.75,
                        'max_aspect_ratio': 3.0,
                        'min_area': 0.75,
                        'max_area': 1.0,
                        'overlap_thresh': 0.0
                    }
                },
                {
                    'random_adjust_brightness': {
                        'max_delta': 0.2
                    }
                },
                {
                    'random_adjust_contrast': {
                        'min_delta': 0.8,
                        'max_delta': 1.25
                    }
                },
                {
                    'random_adjust_saturation': {
                        'min_delta': 0.8,
                        'max_delta': 1.25
                    }
                },
                {
                    'random_adjust_hue': {
                        'max_delta': 0.02
                    }
                }
            ],
            'optimizer': {
                'momentum_optimizer': {
                    'learning_rate': {
                        'cosine_decay_learning_rate': {
                            'learning_rate_base': 0.04,
                            'total_steps': 50000,
                            'warmup_learning_rate': 0.013333,
                            'warmup_steps': 2000
                        }
                    },
                    'momentum_optimizer_value': 0.9
                },
                'use_moving_average': False
            },
            'fine_tune_checkpoint': None,
            'fine_tune_checkpoint_type': 'detection',
            'fine_tune_checkpoint_version': 'V2',
            'num_steps': 50000,
            'startup_delay_steps': 0.0,
            'replicas_to_aggregate': 1,
            'max_number_of_boxes': 100,
            'unpad_groundtruth_tensors': False,
            'use_bfloat16': False
        },
        'train_input_config': {
            'label_map_path': 'label_map.pbtxt',
            'tf_record_input_reader': {
                'input_path': 'train.record'
            }
        },
        'eval_config': {
            'metrics_set': ['coco_detection_metrics'],
            'use_moving_averages': False,
            'batch_size': 1
        },
        'eval_input_config': {
            'label_map_path': 'label_map.pbtxt',
            'shuffle': False,
            'num_epochs': 1,
            'tf_record_input_reader': {
                'input_path': 'val.record'
            }
        }
    }
    
    return config_util.create_config_from_dict(pipeline_config)

def create_label_map(category_index, output_path):
    with open(output_path, 'w') as f:
        for category_id, category_name in category_index.items():
            f.write(f'item {{\n')
            f.write(f'  id: {category_id}\n')
            f.write(f'  name: \'{category_name}\'\n')
            f.write(f'}}\n\n')

def train_model(args):
    # Create label map
    with open(args.category_index_file, 'r') as f:
        category_index = json.load(f)
    create_label_map(category_index, os.path.join(args.output_dir, 'label_map.pbtxt'))
    
    # Create model config
    configs = create_model_config(len(category_index), args.batch_size)
    configs = config_util.merge_external_params_with_configs(configs, None)
    
    # Create model
    model = model_builder.build(model_config=configs['model'], is_training=True)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.04,
        momentum=0.9
    )
    
    # Create checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, args.output_dir, max_to_keep=5
    )
    
    # Create training metrics
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    
    # Training loop
    train_losses = []
    val_maps = []
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss_metric.reset_states()
        for batch in tqdm(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(batch['image'], training=True)
                loss = model.loss(predictions, batch)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss_metric.update_state(loss)
        
        train_loss = train_loss_metric.result().numpy()
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            eval_results = model.evaluate(val_dataset)
            val_maps.append(eval_results['coco_detection_metrics/mAP'])
            print(f"Validation mAP: {eval_results['coco_detection_metrics/mAP']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_manager.save()
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_maps)
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()
    
    # Export model
    export_dir = os.path.join(args.output_dir, 'exported_model')
    tf.saved_model.save(model, export_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_record', type=str, required=True)
    parser.add_argument('--val_record', type=str, required=True)
    parser.add_argument('--category_index_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)

if __name__ == '__main__':
    main() 