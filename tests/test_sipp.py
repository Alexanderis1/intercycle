from intercycle.robot.sipp import plan

ADJ = {"A": {"B": 5.0}, "B": {"A": 5.0, "G": 2.0}, "G": {"B": 2.0}}


def test_waits_for_window_that_fits():
    # goal only safe in [10,12) (too short for 5 s) and [30,40)
    p = plan(ADJ, "A", 0.0, "G", lambda n: [(10, 12), (30, 40)] if n == "G" else None, need_s=5.0)
    assert p is not None and p.nodes == ["A", "B", "G"]
    assert abs(p.goal_time - 30.0) < 1e-9


def test_immediate_when_window_open():
    p = plan(ADJ, "A", 0.0, "G", lambda n: [(0, 100)] if n == "G" else None, need_s=5.0)
    assert abs(p.goal_time - 7.0) < 1e-9


def test_unsafe_intermediate_node_forces_wait():
    # B unsafe until t=20: robot must wait at A and pass B later
    p = plan(ADJ, "A", 0.0, "G", lambda n: [(20, 1e9)] if n == "B" else None)
    assert p.arrive[1] >= 20 and abs(p.goal_time - 22.0) < 1e-9


def test_no_plan_when_never_fits():
    assert plan(ADJ, "A", 0.0, "G", lambda n: [(10, 12)] if n == "G" else None, need_s=5.0, horizon_s=100) is None
