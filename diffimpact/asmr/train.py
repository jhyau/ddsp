import sys, os
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append(".")
import argparse

from absl import logging
import ddsp.training
import gin
import tensorflow as tf

logging.set_verbosity(logging.INFO)

# Parse args
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--model-type", type=str, choices=['diffimpact', 'ddsp-serra', 'ddsp-sin'],
                    help="the desired model architecture")
group.add_argument("--restore-dir", type=str, default=None,
                    help="search directory for restoring latest checkpoint")
parser.add_argument("--train-pattern", type=str, required=True,
                    help="file pattern for matching training clips")
parser.add_argument("--validation-pattern", type=str, required=True,
                    help="file pattern for matching validation clips")
parser.add_argument("--train-steps", type=int, default=9000)
parser.add_argument("--validation-interval", type=int, default=20,
                    help="number of steps between validation error checks")
parser.add_argument("--save-dir", type=str, required=True,
                     help="destination directory for saving checkpoints")
parser.add_argument("--add-visual-feat", action="store_true", help="Adding visual feature information to modal response pipeline")
parser.add_argument("--visual-feat-train-pattern", type=str, help="file pattern for matching training visual features")
parser.add_argument("--visual-feat-val-pattern", type=str, help="file pattern for matching validation visual features")

args = parser.parse_args()

gin.external_configurable(tf.keras.regularizers.L1, module='tf.keras.regularizers')
if args.model_type:
    gin.add_config_file_search_path('gin/')

    # Adding additional paths
    gin.add_config_file_search_path('asmr/gin/')
    gin.add_config_file_search_path('diffimpact/asmr/gin/')
    gin.add_config_file_search_path('ddsp/diffimpact/asmr/gin/')

    gin.parse_config_file('models/%s.gin'%args.model_type)
    gin.parse_config_file('training.gin')
    gin.parse_config_file('asmr.gin')
    restore_dir = args.save_dir
else:
    print("Restoring from checkpoint: ", args.restore_dir)
    # If restoring from checkpoint, use the gin config stored with the checkpoint
    gin.parse_config_file(ddsp.training.train_util.get_latest_operative_config(args.restore_dir))
    restore_dir = args.restore_dir
gin.config.bind_parameter('%TRAIN_FILE_PATTERN', args.train_pattern)
gin.config.bind_parameter('%VALIDATION_FILE_PATTERN', args.validation_pattern)
# Bind the save dir as well
gin.config.bind_parameter('%SAVE_DIR', args.save_dir)

print(gin.config.query_parameter('%TRAIN_FILE_PATTERN'))
print(f"Loaded gin file... \nThe N_MODAL_FREQUENCIES is : {gin.config.query_parameter('%N_MODAL_FREQUENCIES')} ")

# If adding visual features
if args.add_visual_feat:
    gin.config.bind_parameter('%ADD_VISUAL_FEAT', args.add_visual_feat)
    gin.config.bind_parameter('%VISUAL_TRAIN_FILE_PATTERN', args.visual_feat_train_pattern)
    gin.config.bind_parameter('$VISUAL_VAL_FILE_PATTERN', args.visual_feat_val_pattern)

# Set dynamic GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

ddsp.training.train_util.train(data_provider=gin.REQUIRED,
                               trainer=gin.REQUIRED,
                               batch_size=gin.REQUIRED,
                               steps_per_summary=1,
                               num_steps=args.train_steps,
                               save_dir=args.save_dir,
                               restore_dir=restore_dir,
                               validation_steps=args.validation_interval,
                               wandb_logging=False,
                               validation_provider=gin.REQUIRED
                              )
