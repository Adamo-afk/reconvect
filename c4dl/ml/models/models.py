import gc
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Layer
from tensorflow.keras.layers import Activation, ReLU, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import TimeDistributed, Lambda, Add
from tensorflow.keras.optimizers import Adam

from .blocks import ConvBlock, ResBlock
from ...features.batch import BatchSequence
from .optimizers import AdaBeliefOptimizer
from .rnn import ConvGRU, ResGRU


file_dir = os.path.dirname(os.path.abspath(__file__))


def concat(**kwargs):
    # workaround for the behavior in Concatenate 
    # that raises an error if the input list is of length 1
    def concat_func(inputs):
        if len(inputs) > 1:
            return Concatenate(**kwargs)(inputs)
        else:
            return inputs[0]
    return concat_func


from keras.engine import base_preprocessing_layer
class Cast(base_preprocessing_layer.PreprocessingLayer):
    def call(self, x):
        return tf.cast(x, tf.float32)


def rnn_model(
    input_specs,
    base_shape=(256,256),
    past_timesteps=12,
    future_timesteps=12,
    num_outputs=1
):
    # separate inputs by resolution and timeframe; build input list
    inputs_by_shape = {}    
    inputs = []
    def add_input(timeframe, shape_divisor, ip):
        if timeframe not in inputs_by_shape:
            inputs_by_shape[timeframe] = {}
        if shape_divisor not in inputs_by_shape[timeframe]:
            inputs_by_shape[timeframe][shape_divisor] = []
        inputs_by_shape[timeframe][shape_divisor].append(ip)

    for input_spec in input_specs:
        timeframe = input_spec["timeframe"]
        if timeframe == "past":
            timesteps = past_timesteps
        elif timeframe == "future":
            timesteps = future_timesteps
        elif timeframe == "static":
            timesteps = 1
        shape_divisor = input_spec.get("shape_divisor", 1)
        shape = (base_shape[0]//shape_divisor, base_shape[1]//shape_divisor)
        channels = input_spec.get("channels", 1)
        dtype = input_spec.get("dtype", tf.float32)
        
        ip = Input(
            shape=(timesteps,shape[0],shape[1],channels),
            name=input_spec["name"],
            dtype=dtype
        )
        inputs.append(ip)
        if dtype != np.float32:
            ip = tf.cast(ip, tf.float32)
        
        if timeframe == "static": # expand static variable in time dimension
            ip_past = tf.repeat(ip, axis=1, repeats=past_timesteps)
            add_input("past", shape_divisor, ip_past)
            if future_timesteps != past_timesteps:
                ip_future = tf.repeat(ip, axis=1, repeats=future_timesteps)
            else:
                ip_future = ip_past
            add_input("future", shape_divisor, ip_future)
        else:
            add_input(timeframe, shape_divisor, ip)

    # number of channels by depth
    #block_channels = [32, 64, 128, 256]
    block_channels = [32, 64, 128]    
    #block_channels = [24, 48, 96]

    # recurrent downsampling 
    xt_by_time = {}
    for timeframe in inputs_by_shape:
        xt_by_time[timeframe] = {
            s: concat(axis=-1)(inputs_by_shape[timeframe][s])
            for s in inputs_by_shape[timeframe]
        }
    
    intermediate = []
    for timeframe in inputs_by_shape:
        xt = xt_by_time[timeframe]

        for (i,channels) in enumerate(block_channels):
            # merge different resolutions when possible
            s = 2**i
            if (i > 0) and s in inputs_by_shape[timeframe]:
                if 1 in xt:                
                    xt[1] = Concatenate(axis=-1)([xt[1],xt[s]])
                else:
                    xt[1] = xt[s]
                del xt[s]

            for s in xt:
                stride = 2 if (s == 1) else 1 # do not downsample lores data
                xt[s] = ResBlock(channels, time_dist=True, stride=stride)(xt[s])
                
                initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt[s])
                # TODO: future steps should iterate backwards in time?
                
                xt[s] = ResGRU(                
                    channels, return_sequences=True, 
                    time_steps=past_timesteps if timeframe=="past" else future_timesteps,
                )([xt[s],initial_state])
                

            if timeframe == "past":
                intermediate.append(ConvBlock(channels)(xt[1][:,-1,...]))

        xt_by_time[timeframe] = xt[1]

    # recurrent upsampling
    if "future" in xt_by_time:
        xt = xt_by_time["future"]
    else:
        xt = Lambda(lambda y: tf.zeros_like(
            tf.repeat(y[:,:1,...],future_timesteps,axis=1)
        ))(xt_by_time["past"])

    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = ResGRU(        
            channels, return_sequences=True, time_steps=future_timesteps
        )([xt,intermediate[i]])        
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = ResBlock(block_channels[max(i-1,0)], time_dist=True)(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1),
        activation='sigmoid'))(xt)

    model = Model(inputs=inputs, outputs=[seq_out])

    return model


def persistence_model(
    num_inputs=1,
    num_outputs=1,
    past_timesteps=12,
    future_timesteps=12,
    output_names=None
    ):

    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = [past_in]

    last_timestep = past_in[:,-1:,...]

    persistence = tf.repeat(last_timestep, future_timesteps, axis=1)

    outputs = [
        Lambda(lambda x: x, name=name)(persistence[...,i:i+1])
        for (i,name) in enumerate(output_names)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    return model


def logit(x):
    return tf.math.log(x/(1-x))


def dice_coef(y_true, y_pred, smooth=1):
    axes = (1,2,3,4)
    s = lambda x: tf.math.reduce_sum(x, axis=axes)
    y_true = tf.cast(y_true, tf.float32)
    intersection = s(y_true * y_pred)
    return (2. * intersection + smooth) / (s(y_true) + s(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return 1 - dice_coef(y_true, y_pred)


def iou_metric(y_true, y_pred): # this is the same as critical success index
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    intersection = y_true * y_pred
    union = (1 - y_true) * y_pred + y_true
    int_sum = tf.math.reduce_sum(intersection, axis=(1,2,3,4))
    uni_sum = tf.math.reduce_sum(union, axis=(1,2,3,4))
    return tf.where(uni_sum != 0, int_sum / uni_sum, 1)


def dice_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    true_pos = y_true * y_pred
    false_pos = (1 - y_true) * y_pred
    false_neg = y_true * (1 - y_pred)

    tp = tf.math.reduce_sum(true_pos, axis=(1,2,3,4))
    fp = tf.math.reduce_sum(false_pos, axis=(1,2,3,4))
    fn = tf.math.reduce_sum(false_neg, axis=(1,2,3,4))
    denom = 2*tp + fp + fn
    return tf.where(denom != 0, 2*tp / denom, 1)


def true_pos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    tp = y_true * y_pred
    return tf.math.reduce_mean(tp, axis=(1,2,3,4))


def true_neg(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    tn = (1-y_true) * (1-y_pred)
    return tf.math.reduce_mean(tn, axis=(1,2,3,4))


def false_pos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    fp = (1-y_true) * y_pred
    return tf.math.reduce_mean(fp, axis=(1,2,3,4))


def false_neg(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    fn = y_true * (1-y_pred)
    return tf.math.reduce_mean(fn, axis=(1,2,3,4))


def create_weighted_binary_crossentropy(ones_fraction):
    zeros_fraction = 1-ones_fraction
    weights = (
        1./(2*zeros_fraction),
        1./(2*ones_fraction)
    )

    @tf.function
    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        loss = tf.losses.binary_crossentropy(y_true, y_pred)
        # Apply the weights
        w = (1 - y_true) * weights[0] + y_true * weights[1]
        weighted_loss = w[...,0] * loss
        # Return the mean error
        return weighted_loss

    return weighted_binary_crossentropy


def create_weighted_focal_loss(ones_fraction, gamma=tf.constant(2.0)):
    wce = create_weighted_binary_crossentropy(tf.constant(ones_fraction))
    
    def weighted_focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.constant(0.001) + y_pred*tf.constant(0.998) # scale to inhibit exploding gradients
        ce = wce(y_true, y_pred)
        pt = tf.where(y_true==1, y_pred, 1-y_pred)
        return (1-pt[...,0])**gamma * ce
    return weighted_focal_loss


def compile_model(
    model,
    optimizer='adabelief',
    loss='weighted_focal_loss', 
    metrics=[
        'binary_accuracy', "iou_metric", "dice_metric",
        "true_pos", "true_neg", "false_pos", "false_neg"
    ],
    #event_occurrence=0.00278 # R10
    event_occurrence=0.0106 # occurrence-10
):
    metric_names = {
        "weighted_binary_crossentropy": create_weighted_binary_crossentropy(
            event_occurrence),
        "weighted_focal_loss": create_weighted_focal_loss(
            event_occurrence),
        "iou_metric": iou_metric,
        "dice_metric": dice_metric,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg
    }
    loss = metric_names.get(loss, loss)
    metrics = [metric_names.get(m,m) for m in metrics]
    if optimizer == "adabelief":
        optimizer = AdaBeliefOptimizer()
    model.compile(loss=loss,
        optimizer=optimizer, metrics=metrics)


def init_model(batch_gen, model_func=rnn_model, compile=True, 
    init_strategy=True, **kwargs):

    (past_timesteps, future_timesteps) = batch_gen.timesteps
    num_outputs = len(batch_gen.target_names)

    # construct input specs from a sample batch
    input_specs = []
    (X, y) = batch_gen.batch(0)
    max_size = max(x.shape[2] for x in X)
    pred_names = batch_gen.pred_names_past + \
        batch_gen.pred_names_future + \
        batch_gen.pred_names_static

    for (i,x) in enumerate(X):
        shape_divisor = max_size // x.shape[2]
        timesteps = x.shape[1]
        channels = x.shape[-1]
        pred_name = pred_names[i]
        if pred_name in batch_gen.pred_names_past:
            timeframe = "past"
        elif pred_name in batch_gen.pred_names_future:
            timeframe = "future"
        elif pred_name in batch_gen.pred_names_static:
            timeframe = "static"        
        input_spec = {
            "shape_divisor": shape_divisor,
            "channels": channels,
            "timeframe": timeframe,
            "name": pred_name,
            "dtype": x.dtype
        }
        input_specs.append(input_spec)

    if init_strategy and len(tf.config.list_physical_devices('GPU')) > 1:
        # initialize multi-GPU strategy
        strategy = tf.distribute.MirroredStrategy()
    else: # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = model_func(
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            input_specs=input_specs,
            num_outputs=num_outputs,
            **kwargs
        )
        if compile:
            compile_model(model)

    gc.collect()
    
    return (model, strategy)


def combined_model(models, output_names):
    past_in = Input(shape=models[0].input_shape[1:],
        name="past_in")
    outputs = [
        Layer(name=name)(model(past_in))
        for (model, name) in zip(models, output_names)
    ]
    comb_model = Model(inputs=[past_in], outputs=outputs)

    return comb_model


def train_model(model, strategy, batch_gen,
    weight_fn="model.h5", monitor="val_loss"):

    fn = os.path.join(file_dir, "../../../models", weight_fn)
    steps_per_epoch = len(batch_gen.time_coords["train"]) // batch_gen.batch_size
    validation_steps = len(batch_gen.time_coords["valid"]) // batch_gen.batch_size

    with strategy.scope():        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            fn, save_weights_only=True, save_best_only=True, mode="min",
            monitor=monitor
        )
        reducelr = tf.keras.callbacks.ReduceLROnPlateau(
            patience=3, mode="min", factor=0.2, monitor=monitor,
            verbose=1
        )
        earlystop = tf.keras.callbacks.EarlyStopping(
            patience=6, mode="min", restore_best_weights=True,
            monitor=monitor
        )
        callbacks = [checkpoint, reducelr, earlystop]

        batch_seq_train = BatchSequence(batch_gen, dataset='train')
        batch_seq_valid = BatchSequence(batch_gen, dataset='valid')

        model.fit(
            batch_seq_train,
            epochs=100,
            steps_per_epoch=steps_per_epoch,
            validation_data=batch_seq_valid,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
