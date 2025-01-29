# cython: profile=True
import copy
import importlib
import itertools
from typing import Tuple, Dict, Callable

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, sin, cos, floor, atan2, M_PI, abs, asin

# from highway_env.road.lane cimport AbstractLane

np.import_array()

from highway_env.types import Vector, Interval

cdef extern from "c_utils.c":
    char get_rectangle_intersection(float p0_x, float p0_y, float p1_x, float p1_y, float* xs, float* ys, float *t_o)
    void c_rect_corners(float center_x, float center_y, float length, float width, float angle, float *out_x, float *out_y)

cdef distance_to_rect2(np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=1] q, float* xs, float* ys, float max_range):
    cdef float t_o
    cdef char coll =  get_rectangle_intersection(r[0], r[1], q[0], q[1], xs, ys, &t_o)

    if coll:
        return t_o * max_range
    else:
        return np.inf

cdef np.ndarray rect_corners2(np.ndarray[double, ndim=1] center, float length, float width, float angle, float* out_x, float* out_y):
    cdef float center_x = center[0]
    cdef float center_y = center[1]

    c_rect_corners(center_x, center_y, length, width, angle, out_x, out_y)

    return

def argmin(lst):
      return lst.index(min(lst))

def clip(value, low, high):
    return min(max(low, value), high)

def norm(p1, p2):
    # dont use the sqrt to save some time (comparison need to be made against squares)
    # return ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)

    cdef float x1, y1, x2, y2
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    return ((x1 - x2) ** 2) + ((y1 - y2) ** 2)

    # return (
            # ((p1[0] - p2[0]) ** 2) +
            # ((p1[1] - p2[1]) ** 2)
            # ) ** 0.5

cdef inline normalize_vector(np.ndarray[double, ndim=1] vec):
    cdef float length

    length =  sqrt(
                (vec[0] ** 2) +
                (vec[1] ** 2)
              )

    vec[0] /= length
    vec[1] /= length

    return vec

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
# cdef inline float vec_norm(x: float, y:float):
cdef inline float vec_norm(np.ndarray[double, ndim=1] vec):

    cdef float x, y
    x = vec[0]
    y = vec[1]

    return sqrt(
            (x ** 2) +
            (y ** 2)
            )

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline float c_dot(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
    return a[0] * b[0] + a[1] * b[1]

def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def constrain(x: float, a: float, b: float) -> np.ndarray:
    return np.clip(x, a, b)


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    # return ((x + np.pi) % (2 * np.pi)) - np.pi
    return ((x + 3.14) % (2 * 3.14)) - 3.14


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray rect_corners(
    np.ndarray[double, ndim=1] center,
    length: float,
    width: float,
    angle: float,
):
    """
    Returns the positions of the corners of a rectangle.
    :param center: the rectangle center
    :param length: the rectangle length
    :param width: the rectangle width
    :param angle: the rectangle angle
    :return: a list of positions
    """
    cdef np.ndarray[double, ndim=1] half_l, half_w
    cdef np.ndarray[double, ndim=2] corners

    half_l = np.array([length / 2, 0])
    half_w = np.array([0, width / 2])
    corners = np.array([-half_l - half_w, -half_l + half_w, +half_l + half_w, +half_l - half_w])

    # return [np.array(rotate(c[0], c[1], angle)) + center for c in corners]

    cdef float c, s
    c, s = cos(angle), sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    # return (rotation @ corners.T).T + np.tile(center, (len(corners), 1))
    return (rotation.dot(corners.T)).T + np.tile(center, (len(corners), 1))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
# def distance_to_rect(line: Tuple[np.ndarray, np.ndarray], rect: List[np.ndarray]):
def distance_to_rect(np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=1] q, rect: List[np.ndarray]):
#cdef inline normalize_vector(np.ndarray[double, ndim=1] vec):
    """
    Compute the intersection between a line segment and a rectangle.

    See https://math.stackexchange.com/a/2788041.
    :param line: a line segment [R, Q]
    :param rect: a rectangle [A, B, C, D]
    :return: the distance between R and the intersection of the segment RQ with the rectangle ABCD
    """
    cdef np.ndarray a, b, c, d
    cdef np.ndarray u, v, qr
    cdef float rqu, rqv
    cdef float interval_1_1, interval_1_2, interval_2_1, interval_2_2

    a, b, c, d = rect
    u = b - a
    v = d - a
    u = normalize_vector(u)
    v = normalize_vector(v)

    qr = q - r
    rqu = c_dot(qr, u)
    rqv = c_dot(qr, v)
    interval_1_1 = c_dot(a - r, u) / rqu
    interval_1_2 = c_dot(b - r, u) / rqu
    interval_2_1 = c_dot(a - r, v) / rqv
    interval_2_2 = c_dot(d - r, v) / rqv

    if rqu < 0:
        interval_1_1, interval_1_2 = interval_1_2, interval_1_1
    if rqv < 0:
        interval_2_1, interval_2_2 = interval_2_2, interval_2_1

    if (
        interval_distance(interval_1_1, interval_1_2, interval_2_1, interval_2_2) <= 0
        and interval_distance(0, 1, interval_1_1, interval_1_2) <= 0
        and interval_distance(0, 1, interval_2_1, interval_2_2) <= 0
    ):
        return max(interval_1_1, interval_2_1) * vec_norm(qr)
    else:
        return np.inf

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.cdivision(True)
def trace(np.ndarray[double, ndim=1] origin, np.ndarray[double, ndim=1] origin_velocity, float maximum_range, int cells, float angle_increment, objects: list, observer_vehicle) -> np.ndarray:

    cdef float center_angle, distance
    cdef int center_index
    cdef float min_angle, max_angle
    cdef int start, end, index

    cdef int num_radars, cells_per_radar
    num_radars = 4
    cells_per_radar = cells / num_radars

    cdef float[4] corners_x, corners_y

    cdef np.ndarray[double, ndim=2] grid

    grid = np.ones((cells, 3)) * maximum_range
    grid[:, 1] = 0
    grid[:, 2] = -1

    cdef double [:,:] grid_view = grid

    for obstacle in objects:
        if obstacle is observer_vehicle:
            continue
        center_distance = vec_norm(obstacle.position - origin)
        if center_distance > maximum_range:
            continue
        center_angle = position_to_angle(obstacle.position[0], obstacle.position[1], origin, angle_increment)
        center_index = angle_to_index(center_angle, angle_increment, cells)
        distance = center_distance - obstacle.WIDTH / 2
        if distance <= grid_view[center_index, 0]:
            direction = index_to_direction(center_index, angle_increment)
            velocity = c_dot(obstacle.velocity - origin_velocity, direction)
            grid_view[center_index, 0] = distance
            grid_view[center_index, 1] = velocity
            grid_view[center_index, 2] = obstacle.id

        # Angular sector covered by the obstacle
        rect_corners2(obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading, corners_x, corners_y)

        angles = []
        for i in range(4):
            angles.append(position_to_angle(corners_x[i], corners_y[i], origin, angle_increment))

        min_angle, max_angle = min(angles), max(angles)

        if (min_angle < -M_PI / 2 < M_PI / 2 < max_angle):  # Object's corners are wrapping around +pi
            min_angle, max_angle = max_angle, min_angle + 2 * M_PI

        start, end = angle_to_index(min_angle, angle_increment, cells), angle_to_index(max_angle, angle_increment, cells)

        # Actual distance computation for these sections
        for index in range(start, end+1):
            direction = index_to_direction(index, angle_increment)
            distance = distance_to_rect2(origin, origin + maximum_range * direction, corners_x, corners_y, maximum_range)
            if distance <= grid_view[index, 0]:
                velocity = c_dot(obstacle.velocity - origin_velocity, direction)
                grid_view[index, 0] = distance
                grid_view[index, 1] = velocity
                grid_view[index, 2] = obstacle.id
    return grid

cdef inline float position_to_angle(float x, float y, origin: np.ndarray, angle: float):
    return (
        atan2(y - origin[1], x - origin[0])
        + angle / 2
    )

cdef inline int angle_to_index(float angle, float angle_increment, int cells):
    # cant use c expressions here as this leads to wrong results
    # return <int>floor(angle / angle_increment) % cells
    return int(np.floor(angle / angle_increment)) % cells

cdef inline np.ndarray index_to_direction(int index, float angle_increment):
    return np.array([cos(index * angle_increment), sin(index * angle_increment)])


cdef float interval_distance(float min_a, float max_a, float min_b, float max_b):
    """
    Calculate the distance between [minA, maxA] and [minB, maxB]
    The distance will be negative if the intervals overlap
    """
    return min_b - max_a if min_a < min_b else min_a - max_b



def point_in_rectangle(point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
    """
    Check if a point is inside a rectangle

    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float) \
        -> bool:
    """
    Check if a point is inside a rotated rectangle

    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    :return: is the point inside the rectangle
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length/2, -width/2), (length/2, width/2))


def point_in_ellipse(point: Vector, center: Vector, angle: float, length: float, width: float) -> bool:
    """
    Check if a point is inside an ellipse

    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    :return: is the point inside the ellipse
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1: Tuple[Vector, float, float, float],
                                 rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    Do two rotated rectangles intersect?

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    :return: do they?
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1: Tuple[Vector, float, float, float],
                      rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    Check if rect1 has a corner inside rect2

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1/2, 0])
    w1v = np.array([0, w1/2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1+np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def confidence_ellipsoid(data: Dict[str, np.ndarray], lambda_: float = 1e-5, delta: float = 0.1, sigma: float = 0.1,
                         param_bound: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a confidence ellipsoid over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param lambda_: l2 regularization parameter
    :param delta: confidence level
    :param sigma: noise covariance
    :param param_bound: an upper-bound on the parameter norm
    :return: estimated theta, Gramian matrix G_N_lambda, radius beta_N
    """
    phi = np.array(data["features"])
    y = np.array(data["outputs"])
    g_n_lambda = 1/sigma * np.transpose(phi) @ phi + lambda_ * np.identity(phi.shape[-1])
    theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
    d = theta_n_lambda.shape[0]
    beta_n = np.sqrt(2*np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / delta)) + \
        np.sqrt(lambda_*d) * param_bound
    return theta_n_lambda, g_n_lambda, beta_n


def confidence_polytope(data: dict, parameter_box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute a confidence polytope over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: estimated theta, polytope vertices, Gramian matrix G_N_lambda, radius beta_N
    """
    param_bound = np.amax(np.abs(parameter_box))
    theta_n_lambda, g_n_lambda, beta_n = confidence_ellipsoid(data, param_bound=param_bound)

    values, pp = np.linalg.eig(g_n_lambda)
    radius_matrix = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
    h = np.array(list(itertools.product([-1, 1], repeat=theta_n_lambda.shape[0])))
    d_theta = np.array([radius_matrix @ h_k for h_k in h])

    # Clip the parameter and confidence region within the prior parameter box.
    theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
    for k, _ in enumerate(d_theta):
        d_theta[k] = np.clip(d_theta[k], parameter_box[0] - theta_n_lambda, parameter_box[1] - theta_n_lambda)
    return theta_n_lambda, d_theta, g_n_lambda, beta_n


def is_valid_observation(y: np.ndarray, phi: np.ndarray, theta: np.ndarray, gramian: np.ndarray,
                         beta: float, sigma: float = 0.1) -> bool:
    """
    Check if a new observation (phi, y) is valid according to a confidence ellipsoid on theta.

    :param y: observation
    :param phi: feature
    :param theta: estimated parameter
    :param gramian: Gramian matrix
    :param beta: ellipsoid radius
    :param sigma: noise covariance
    :return: validity of the observation
    """
    y_hat = np.tensordot(theta, phi, axes=[0, 0])
    error = np.linalg.norm(y - y_hat)
    eig_phi, _ = np.linalg.eig(phi.transpose() @ phi)
    eig_g, _ = np.linalg.eig(gramian)
    error_bound = np.sqrt(np.amax(eig_phi) / np.amin(eig_g)) * beta + sigma
    return error < error_bound


def is_consistent_dataset(data: dict, parameter_box: np.ndarray = None) -> bool:
    """
    Check whether a dataset {phi_n, y_n} is consistent

    The last observation should be in the confidence ellipsoid obtained by the N-1 first observations.

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: consistency of the dataset
    """
    train_set = copy.deepcopy(data)
    y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
    y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    if train_set["outputs"] and train_set["features"]:
        theta, _, gramian, beta = confidence_polytope(train_set, parameter_box=parameter_box)
        return is_valid_observation(y, phi, theta, gramian, beta)
    else:
        return True
###
# New SAT approach
###
# from math import sqrt
import math

def normalize(vector):
    """
    :return: The vector scaled to a length of 1
    """
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    return vector[0] / norm, vector[1] / norm


def dot(vector1, vector2):
    """
    :return: The dot (or scalar) product of the two vectors
    """
    return vector1[0] * vector2[0] + vector1[1] * vector2[1]


def edge_direction(point0, point1):
    """
    :return: A vector going from point0 to point1
    """
    return point1[0] - point0[0], point1[1] - point0[1]


def orthogonal(vector):
    """
    :return: A new vector which is orthogonal to the given vector
    """
    return vector[1], -vector[0]


def vertices_to_edges(vertices):
    """
    :return: A list of the edges of the vertices as vectors
    """
    return [edge_direction(vertices[i], vertices[(i + 1) % len(vertices)])
            for i in range(len(vertices))]


def project(vertices, axis):
    """
    :return: A vector showing how much of the vertices lies along the axis
    """
    dots = [dot(vertex, axis) for vertex in vertices]
    return [min(dots), max(dots)]


def overlap(projection1, projection2):
    """
    :return: Boolean indicating if the two projections overlap
    """
    return min(projection1) <= max(projection2) and \
           min(projection2) <= max(projection1)


def separating_axis_theorem(vertices_a, vertices_b):
    edges = vertices_to_edges(vertices_a) + vertices_to_edges(vertices_b)
    axes = [normalize(orthogonal(edge)) for edge in edges]

    for axis in axes:
        projection_a = project(vertices_a, axis)
        projection_b = project(vertices_b, axis)

        overlapping = overlap(projection_a, projection_b)

        if not overlapping:
            return False

    return True

def middle_to_vertices(middle, length, width, angle):
    # convert the old represantation of the rectangle to vertices
    angle -= np.deg2rad(90)

    u = np.array([width/2. * np.cos(angle), width/2. * np.sin(angle)])
    v = np.array([-length/2. * np.sin(angle), length/2. * np.cos(angle)])

    a = middle - u + v
    b = middle + u + v
    c = middle + u - v
    d = middle - u - v

    return [a, b, c, d]

import cython
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_closest_lane(list lane_indices, list lanes, np.ndarray  position, heading):
    # return min(lane_indices, key=lambda l:self.get_lane(l).distance_with_heading(position, float(heading)))

   cdef float current_smallest = 1e8
   cdef Tuple [str, str, int] closest_lane_index
   cdef float curr_dist
   cdef int i, l_len

   l_len = len(lane_indices)

   # for index, lane in lane_indices:
   for i in range(l_len):
       # index, lane = lane_indices[i]
       index = lane_indices[i]
       lane = lanes[i]
       curr_dist = lane.distance_with_heading(position, heading)
       if  curr_dist < current_smallest:
           current_smallest = curr_dist
           closest_lane_index = index

   return closest_lane_index

