import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def build_gradcam_models(model):
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))

    last_conv_name = next(
        l.name for l in reversed(base.layers)
        if isinstance(l, tf.keras.layers.Conv2D)
    )
    conv_model = tf.keras.Model(base.input, base.get_layer(last_conv_name).output)

    base_idx = model.layers.index(base)
    head_layers = model.layers[base_idx + 1:]

    inp = tf.keras.Input(shape=conv_model.output_shape[1:])
    x = inp
    for layer in head_layers[:-1]:
        x = layer(x)
    pre_model = tf.keras.Model(inp, x)

    W, b = head_layers[-1].get_weights()
    return conv_model, pre_model, tf.constant(W, tf.float32), tf.constant(b, tf.float32)


def compute_heatmap(img_arr, conv_model, pre_model, W, b):

    img = tf.cast(img_arr, tf.float32)

    with tf.GradientTape() as tape:
        conv_out = tf.cast(conv_model(img), tf.float32)
        tape.watch(conv_out)
        feats  = tf.cast(pre_model(conv_out), tf.float32)
        logits = feats @ W + b
        top_cls = logits[:, tf.argmax(logits[0])]

    grads   = tape.gradient(top_cls, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat    = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
    heat    = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)

    return cv2.resize(heat.numpy().astype(np.float32), IMG_SIZE)


def apply_heatmap(original_rgb, heatmap):
  
    heat_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay  = cv2.addWeighted(
        np.asarray(original_rgb, dtype=np.uint8), 0.55,
        heat_rgb, 0.45, 0
    )
    return heat_rgb, overlay
