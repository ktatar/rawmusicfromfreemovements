# quaternion math implemented using tensorflow tensors
# code adapted from original pyTorch implementation of Quaternet 
# quaternet / common / quaternion.py

import tensorflow as tf
import numpy as np

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) s
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions
    Returns q*r as a tensor of shape (*, 4)
    """

    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape
    
    # Compute outer product
    terms = tf.linalg.matmul(tf.reshape(r, [-1, 4, 1]), tf.reshape(q, [-1, 1, 4]))
    
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return tf.reshape(tf.stack([w, x, y, z], axis=1), original_shape)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = tf.reshape(q, [-1, 4])
    v = tf.reshape(v, [-1, 3])
    
    qvec = q[:, 1:]
    
    uv = tf.linalg.cross(qvec, v)
    uuv = tf.linalg.cross(qvec, uv)
    return tf.reshape((v + 2 * (q[:, :1] * uv + uuv)), original_shape)

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = tf.reshape(q, [-1, 4])
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = tf.math.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = tf.math.asin(tf.clip_by_value(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = tf.math.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = tf.math.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = tf.math.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = tf.math.asin(tf.clip_by_value(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = tf.math.asin(tf.clip_by_value(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = tf.math.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = tf.math.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = tf.math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = tf.math.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = tf.math.asin(tf.clip_by_value(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = tf.math.asin(tf.clip_by_value(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = tf.math.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = tf.math.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = tf.math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = tf.math.asin(tf.clip_by_value(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = tf.math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return tf.reshape(tf.stack((x, y, z), axis=1), original_shape)

# Numpy-backed implementations

def qnormalize_np(q):
    q_orig_shape = q.shape
    q = np.reshape(q, (-1, 4))
    norm = np.linalg.norm(q, axis=1) + 0.000001
    norm = np.reshape(norm, (-1, 1))

    q_norm = q / norm    
    q_norm = np.reshape(q_norm, q_orig_shape)
    
    return q_norm

def qmul_np(q, r):
    q = tf.convert_to_tensor(q)
    r = tf.convert_to_tensor(r)
    return qmul(q, r).numpy()

def qrot_np(q, v):
    q = tf.convert_to_tensor(q)
    v = tf.convert_to_tensor(v)
    return qrot(q, v).numpy()

def qeuler_np(q, order, epsilon=0):
    q = tf.convert_to_tensor(q)
    return qeuler(q, order, epsilon).numpy()

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)
            
    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1
    
    return result.reshape(original_shape)

def slerp(q0, q1, amount=0.5):
    # TODO: modify this code so it works for arrays of quaternions
    
    # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
    q0 = qnormalize_np(q0)
    q1 = qnormalize_np(q1)

    amount = np.clip(amount, 0, 1)

    dot = np.dot(q0, q1)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
    # Fix by reversing one quaternion
    if dot < 0.0:
        q0 = -q0
        dot = -dot

    # sin_theta_0 can not be zero
    if dot > 0.9995:
        qr = q0 + amount * (q1 - q0)
        qr = qnormalize_np(qr)
        return qr

    theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = np.sin(theta_0)
        
    theta = theta_0 * amount
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = (s0 * q0) + (s1 * q1)
    qr = qnormalize_np(qr)
    return qr
