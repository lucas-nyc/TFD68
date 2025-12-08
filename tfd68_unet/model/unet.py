import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose,
    concatenate, Dropout, GlobalAveragePooling2D, Dense,
    Reshape, Multiply, Add, Activation, GlobalMaxPooling2D, Concatenate
)
from tensorflow.keras.models import Model
import config.config as config
from utils.loss import NormalizedMeanError, wing_loss

def cbam_block(input_feature, ratio: int = 8, name: str = None):
    channel = input_feature.shape[-1]
    cbam_name = (name + "_cbam") if name else None
    shared_dense_one = tf.keras.layers.Dense(channel // ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             name=(cbam_name + "_ch_dense1") if cbam_name else None)
    shared_dense_two = tf.keras.layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             name=(cbam_name + "_ch_dense2") if cbam_name else None)

    avg_pool = GlobalAveragePooling2D(name=(cbam_name + "_gap") if cbam_name else None)(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_dense = shared_dense_one(avg_pool)
    avg_dense = shared_dense_two(avg_dense)

    max_pool = GlobalMaxPooling2D(name=(cbam_name + "_gmp") if cbam_name else None)(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_dense = shared_dense_one(max_pool)
    max_dense = shared_dense_two(max_dense)

    ch_sum = Add()([avg_dense, max_dense])
    ch_attention = Activation('sigmoid', name=(cbam_name + "_ch_sigmoid") if cbam_name else None)(ch_sum)
    ch_mul = Multiply()([input_feature, ch_attention])

    avg_pool_sp = tf.reduce_mean(ch_mul, axis=-1, keepdims=True)
    max_pool_sp = tf.reduce_max(ch_mul, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool_sp, max_pool_sp]) 

    spatial = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid',
                     kernel_initializer='he_normal', name=(cbam_name + "_sp_conv") if cbam_name else None)(concat)
    out = Multiply()([ch_mul, spatial])
    return out


def categorical_focal_loss(gamma: float = 2.0, alpha=None, from_logits: bool = False):
    def loss_fn(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        if from_logits:
            y_pred_prob = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred_prob = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true_f * tf.math.log(y_pred_prob), axis=-1) 
        p_t = tf.reduce_sum(y_true_f * y_pred_prob, axis=-1)
        focal_factor = tf.pow(1.0 - p_t, gamma)

        if alpha is None:
            alpha_factor = 1.0
        else:
            a = tf.constant(alpha, dtype=tf.float32)
            alpha_factor = tf.reduce_sum(y_true_f * a, axis=-1)

        loss = alpha_factor * focal_factor * ce
        return loss

    return loss_fn

def build_unet_trunk(input_shape=(256, 256, 1), dropout_rate=config.DROP_OUT):
    inputs = Input(shape=input_shape, name="input_image")

    # encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c3)
    p3 = MaxPooling2D(2)(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c4)
    p4 = MaxPooling2D(2)(c4)

    # bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c5)
    bottleneck = Dropout(dropout_rate, name="bottleneck_dropout")(c5)

    # decoder
    u6 = Conv2DTranspose(512, 3, strides=2, activation='relu', padding='same', kernel_initializer='glorot_normal')(bottleneck)
    m6 = concatenate([c4, u6], axis=3)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(m6)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c6)

    u7 = Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same', kernel_initializer='glorot_normal')(c6)
    m7 = concatenate([c3, u7], axis=3)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(m7)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c7)

    u8 = Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same', kernel_initializer='glorot_normal')(c7)
    m8 = concatenate([c2, u8], axis=3)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(m8)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c8)

    u9 = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same', kernel_initializer='glorot_normal')(c8)
    m9 = concatenate([c1, u9], axis=3)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(m9)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(c9)

    decoder_features = Dropout(dropout_rate, name="decoder_features")(c9)

    trunk = Model(inputs, [decoder_features, bottleneck], name="unet_trunk")
    return trunk

def build_mask_model(trunk: Model, base_lr=config.BASE_LR):
    dec_feat, _ = trunk.output
    mask_output = Conv2D(1, 1, activation='sigmoid', padding='same', name='mask_output')(dec_feat)
    mask_model = Model(trunk.input, mask_output, name='mask_model')
    mask_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=base_lr),
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryIoU(threshold=0.5, name='iou')]
    )
    return mask_model

def build_landmark_model(trunk: Model, num_landmarks: int = config.KEYPOINTS, base_lr=config.BASE_LR, compile_model: bool = True):
    dec_feat, _ = trunk.output
    lm_gap = GlobalAveragePooling2D(name='lm_gap')(dec_feat)
    fc0 = Dense(2048, activation='relu', name='lm_fc0')(lm_gap)
    drop0 = Dropout(0.3, name='lm_drop0')(fc0)
    fc1 = Dense(512, activation='relu', name='lm_fc1')(drop0)
    drop1 = Dropout(0.3, name='lm_drop1')(fc1)
    landmark_output = Dense(num_landmarks * 2, activation='sigmoid', name='landmark_output')(drop1)

    model = Model(trunk.input, landmark_output, name='landmark_model')
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=base_lr),
            loss=wing_loss,
            metrics=[NormalizedMeanError(name='nme')]
        )
    return model

def build_expression_model(trunk: Model,
                           num_classes: int = config.NUM_CLASSES,
                           base_lr: float = config.BASE_LR,
                           use_cbam: bool = False,
                           cbam_ratio: int = 8,
                           focal_params: dict = None,
                           loss_type: str = 'sparse_categorical_crossentropy',
                           compile_model: bool = True):
    dec_feat, bottleneck = trunk.output

    b_feat = bottleneck
    d_feat = dec_feat
    if use_cbam:
        try:
            b_feat = cbam_block(b_feat, ratio=cbam_ratio, name="cbam_bottleneck")
            d_feat = cbam_block(d_feat, ratio=cbam_ratio, name="cbam_decoder")
        except Exception:
            b_feat = bottleneck
            d_feat = dec_feat

    gap_bottleneck = GlobalAveragePooling2D(name='gap_bottleneck')(b_feat)
    gap_decoder = GlobalAveragePooling2D(name='gap_decoder')(d_feat)
    merged_gap = keras.layers.concatenate([gap_bottleneck, gap_decoder], name='combined_gap')

    x = Dense(256, activation='relu', name='expr_fc1')(merged_gap)
    x = Dropout(0.5, name='expr_dropout')(x)

    if loss_type == 'binary_crossentropy':
        expr_activation = 'sigmoid'
        loss_expr = 'binary_crossentropy'
    elif loss_type == 'categorical_crossentropy':
        expr_activation = 'softmax'
        loss_expr = 'categorical_crossentropy'
    else:
        expr_activation = 'softmax'
        loss_expr = 'sparse_categorical_crossentropy'

    expression_output = Dense(num_classes, activation=expr_activation, name='expression_output')(x)

    model = Model(trunk.input, expression_output, name='expression_model')

    if compile_model:
        if focal_params:
            gamma = float(focal_params.get('gamma', 2.0))
            alpha = focal_params.get('alpha', None)
            from_logits = bool(focal_params.get('from_logits', False))
            fl = categorical_focal_loss(gamma=gamma, alpha=alpha, from_logits=from_logits)
            loss_to_use = fl
        else:
            loss_to_use = loss_expr

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=base_lr),
            loss=loss_to_use,
            metrics=[keras.metrics.CategoricalAccuracy(name='expr_acc')] if expr_activation == 'softmax' else [keras.metrics.BinaryAccuracy(name='expr_acc')],
            weighted_metrics={'expression_output': []}
        )
    return model


def build_final_model(trunk: Model,
                      num_landmarks: int = config.KEYPOINTS,
                      num_classes: int = config.NUM_CLASSES,
                      base_lr: float = config.BASE_LR,
                      use_cbam: bool = False,
                      cbam_ratio: int = 8,
                      focal_params: dict = None,
                      loss_type: str = 'sparse_categorical_crossentropy',
                      compile_model: bool = True):
    dec_feat, bottleneck = trunk.output

    lm_gap = GlobalAveragePooling2D(name='lm_gap')(dec_feat)
    fc0 = Dense(2048, activation='relu', name='lm_fc0')(lm_gap)
    drop0 = Dropout(0.3, name='lm_drop0')(fc0)
    fc1 = Dense(512, activation='relu', name='lm_fc1')(drop0)
    drop1 = Dropout(0.3, name='lm_drop1')(fc1)
    landmark_output = Dense(num_landmarks * 2, activation='sigmoid', name='landmark_output')(drop1)

    b_feat = bottleneck
    d_feat = dec_feat
    if use_cbam:
        try:
            b_feat = cbam_block(b_feat, ratio=cbam_ratio, name="cbam_bottleneck")
            d_feat = cbam_block(d_feat, ratio=cbam_ratio, name="cbam_decoder")
        except Exception:
            b_feat = bottleneck
            d_feat = dec_feat

    gap_bottleneck = GlobalAveragePooling2D(name='gap_bottleneck')(b_feat)
    gap_decoder = GlobalAveragePooling2D(name='gap_decoder')(d_feat)
    merged_gap = keras.layers.concatenate([gap_bottleneck, gap_decoder], name='combined_gap')

    x = Dense(256, activation='relu', name='expr_fc1')(merged_gap)
    x = Dropout(0.5, name='expr_dropout')(x)

    if loss_type == 'binary_crossentropy':
        expr_activation = 'sigmoid'
        loss_expr = 'binary_crossentropy'
    elif loss_type == 'categorical_crossentropy':
        expr_activation = 'softmax'
        loss_expr = 'categorical_crossentropy'
    else:
        expr_activation = 'softmax'
        loss_expr = 'sparse_categorical_crossentropy'

    expression_output = Dense(num_classes, activation=expr_activation, name='expression_output')(x)

    final_model = Model(trunk.input, [landmark_output, expression_output], name='unet_final_2heads_cbam')

    if compile_model:
        if focal_params:
            gamma = float(focal_params.get('gamma', 2.0))
            alpha = focal_params.get('alpha', None)
            from_logits = bool(focal_params.get('from_logits', False))
            fl = categorical_focal_loss(gamma=gamma, alpha=alpha, from_logits=from_logits)
            loss_expr_to_use = fl
        else:
            loss_expr_to_use = loss_expr

        final_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=base_lr),
            loss={
                'landmark_output': wing_loss,
                'expression_output': loss_expr_to_use
            },
            loss_weights={
                'landmark_output': 1.0,
                'expression_output': 0.5
            },
            metrics={
                'landmark_output': [NormalizedMeanError(name='nme')],
                'expression_output': [keras.metrics.CategoricalAccuracy(name='expr_acc')] if expr_activation == 'softmax' else [keras.metrics.BinaryAccuracy(name='expr_acc')]
            },
            weighted_metrics={
                'expression_output': [keras.metrics.CategoricalAccuracy(name='expr_w_acc')] if expr_activation == 'softmax' else [keras.metrics.BinaryAccuracy(name='expr_w_acc')]
            }
        )

    return final_model

