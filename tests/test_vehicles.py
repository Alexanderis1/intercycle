import numpy as np

from intercycle.perception.vehicles import foot_points, suppress_people_on_forklifts


def test_foot_point_is_bottom_centre():
    assert np.allclose(foot_points(np.array([[10, 20, 30, 100]])), [[20, 100]])


def test_driver_inside_forklift_is_suppressed_walker_kept():
    forklift = np.array([[100, 100, 300, 400]])
    people = np.array([
        [150, 120, 210, 300],  # driver: foot (180, 300) inside the forklift box
        [400, 100, 460, 380],  # walker beside it: foot (430, 380) outside
        [80, 50, 140, 250],    # box overlaps the forklift but foot (110, 250) is inside -> suppressed
        [60, 150, 120, 410],   # box overlaps but foot (90, 410) is below/left of the forklift -> kept
    ])
    assert suppress_people_on_forklifts(people, forklift).tolist() == [False, True, False, True]


def test_border_counts_as_inside():
    assert suppress_people_on_forklifts(np.array([[90, 0, 110, 400]]), np.array([[100, 100, 300, 400]])).tolist() == [False]


def test_empty_inputs():
    assert suppress_people_on_forklifts(np.zeros((2, 4)), np.zeros((0, 4))).tolist() == [True, True]
    assert suppress_people_on_forklifts(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).tolist() == []


def test_any_of_several_forklifts():
    fks = np.array([[0, 0, 10, 10], [100, 100, 200, 200]])
    people = np.array([[140, 50, 160, 150], [40, 0, 60, 50]])
    assert suppress_people_on_forklifts(people, fks).tolist() == [False, True]
