import tensorflow as tf
from tensorflow import keras
import config.config as config
try:
    import tfd68_unet.config.config as config
except Exception:
    import config.config as config

class NormalizedMeanError(keras.metrics.Metric):
    def __init__(self, name='nme', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_nme = self.add_weight(name='sum_nme', initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        batch = tf.shape(y_true)[0]
        flat_dim = tf.shape(y_true)[1]
        K = flat_dim // 2

        y_true_r = tf.reshape(y_true, (batch, K, 2)) 
        y_pred_r = tf.reshape(y_pred, (batch, K, 2))

        dists = tf.norm(y_true_r - y_pred_r, axis=2) 
        per_sample_mean = tf.reduce_mean(dists, axis=1)  
        nonzero_mask = tf.reduce_any(tf.not_equal(y_true_r, 0.0), axis=2) 
        RIGHT_OUTER_IDX = 36
        LEFT_OUTER_IDX = 45

        def safe_gather(idx):
            def _gather():
                return y_true_r[:, idx, :]
            def _zeros():
                return tf.zeros((batch, 2), dtype=y_true_r.dtype)
            return tf.cond(tf.greater(idx, K-1), _zeros, _gather)

        cond_outer_exist = tf.logical_and(tf.greater(K, RIGHT_OUTER_IDX), tf.greater(K, LEFT_OUTER_IDX))
        iod_outer = tf.cond(cond_outer_exist,
                            lambda: tf.norm(y_true_r[:, RIGHT_OUTER_IDX, :] - y_true_r[:, LEFT_OUTER_IDX, :], axis=1),
                            lambda: tf.zeros((batch,), dtype=tf.float32))

        cond_eye_indices_exist = tf.greater_equal(K, 48)
        iod_eye = tf.zeros((batch,), dtype=tf.float32)
        eye_counts_r = tf.zeros((batch,), dtype=tf.float32)
        eye_counts_l = tf.zeros((batch,), dtype=tf.float32)
        if cond_eye_indices_exist:
            right_slice = y_true_r[:, 36:42, :]
            left_slice = y_true_r[:, 42:48, :]
            right_mask = nonzero_mask[:, 36:42]
            left_mask = nonzero_mask[:, 42:48]

            def masked_center(coords, mask):
                maskf = tf.cast(mask, tf.float32)
                maskf = tf.expand_dims(maskf, axis=2)
                sums = tf.reduce_sum(coords * maskf, axis=1)
                counts = tf.reduce_sum(maskf, axis=1)
                counts_safe = tf.where(tf.equal(counts, 0.0), tf.ones_like(counts), counts)
                return sums / counts_safe

            right_center = masked_center(right_slice, right_mask)
            left_center = masked_center(left_slice, left_mask)
            iod_eye_tmp = tf.norm(right_center - left_center, axis=1)
            iod_eye = iod_eye_tmp
            eye_counts_r = tf.reduce_sum(tf.cast(right_mask, tf.float32), axis=1)
            eye_counts_l = tf.reduce_sum(tf.cast(left_mask, tf.float32), axis=1)

        coords_x = y_true_r[:, :, 0]
        coords_y = y_true_r[:, :, 1]
        inf = tf.constant(1e12, dtype=tf.float32)
        neg_inf = -inf
        min_x = tf.reduce_min(tf.where(nonzero_mask, coords_x, tf.fill(tf.shape(coords_x), inf)), axis=1)
        max_x = tf.reduce_max(tf.where(nonzero_mask, coords_x, tf.fill(tf.shape(coords_x), neg_inf)), axis=1)
        min_y = tf.reduce_min(tf.where(nonzero_mask, coords_y, tf.fill(tf.shape(coords_y), inf)), axis=1)
        max_y = tf.reduce_max(tf.where(nonzero_mask, coords_y, tf.fill(tf.shape(coords_y), neg_inf)), axis=1)
        no_points = tf.logical_or(tf.math.is_inf(min_x), tf.math.is_inf(min_y))
        bbox_diag = tf.sqrt(tf.square(max_x - min_x) + tf.square(max_y - min_y))
        bbox_diag = tf.where(no_points, tf.zeros_like(bbox_diag), bbox_diag)

        cand_outer = tf.where(tf.logical_and(cond_outer_exist, tf.logical_and(nonzero_mask[:, RIGHT_OUTER_IDX], nonzero_mask[:, LEFT_OUTER_IDX])), iod_outer, tf.zeros_like(iod_outer))
        eye_valid = tf.logical_and(tf.greater(eye_counts_r, 0.0), tf.greater(eye_counts_l, 0.0))
        cand_eye = tf.where(cond_eye_indices_exist & eye_valid, iod_eye, tf.zeros_like(iod_outer))
        cand_bbox = bbox_diag

        stacked = tf.stack([cand_outer, cand_eye, cand_bbox], axis=1)
        iod = tf.reduce_max(stacked, axis=1)
        iod = tf.where(iod > 1e-6, iod, tf.fill(tf.shape(iod), 1e-6))

        nme_per_sample = per_sample_mean / iod

        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            sw = tf.reshape(sw, (-1,))
            batch_sum = tf.reduce_sum(nme_per_sample * sw)
            weight_sum = tf.reduce_sum(sw)
            self.sum_nme.assign_add(batch_sum)
            self.count.assign_add(weight_sum)
        else:
            self.sum_nme.assign_add(tf.reduce_sum(nme_per_sample))
            self.count.assign_add(tf.cast(batch, tf.float32))

    def result(self):
        return self.sum_nme / (self.count + 1e-12)

    def reset_state(self):
        self.sum_nme.assign(0.0)
        self.count.assign(0.0)


def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    x = y_true - y_pred
    abs_x = tf.abs(x)
    C = w * (1.0 - tf.math.log(1.0 + w / epsilon))
    small_mask = tf.less(abs_x, w)
    small_loss = w * tf.math.log(1.0 + abs_x / epsilon)
    large_loss = abs_x - C
    losses = tf.where(small_mask, small_loss, large_loss)
    return tf.reduce_mean(losses)

def wing_loss_on_normalized(y_true, y_pred, w_pixels=10.0, epsilon_pixels=2.0):
    out_w = float(getattr(config, "OUT_W", 256))
    out_h = float(getattr(config, "OUT_H", 256))
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    batch = tf.shape(y_true)[0]
    flat_dim = tf.shape(y_true)[1]
    K = flat_dim // 2
    y_true_r = tf.reshape(y_true, (batch, K, 2))
    y_pred_r = tf.reshape(y_pred, (batch, K, 2))
    scale = tf.constant([out_w, out_h], dtype=tf.float32)
    y_true_px = y_true_r * scale
    y_pred_px = y_pred_r * scale
    y_true_px_flat = tf.reshape(y_true_px, (batch, K * 2))
    y_pred_px_flat = tf.reshape(y_pred_px, (batch, K * 2))
    return wing_loss(y_true_px_flat, y_pred_px_flat, w=w_pixels, epsilon=epsilon_pixels)
