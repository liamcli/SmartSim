import sys
import warnings

import pytest

from smartsim import Experiment, status

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


@pytest.mark.parametrize("db_type", ["uds", "tcp", "deprecated"])
def test_launch_colocated_model(fileutils, wlmutils, db_type):
    """Test the launch of a model with a colocated database"""

    launcher = wlmutils.get_test_launcher()
    exp_name = "test-launch-colocated-model-with-restart"
    exp = Experiment(exp_name, launcher=launcher)

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("send_data_local_smartredis.py")
    network_interface = wlmutils.get_test_interface()

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)
    colo_settings.set_tasks_per_node(1)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    db_args = {
        "db_cpus": 1,
        "limit_app_cpus": False,
        "debug": True,
    }

    if db_type in ["tcp", "deprecated"]:
        colocate_fun = {
            "tcp": colo_model.colocate_db_tcp,
            "deprecated": colo_model.colocate_db,
        }
        with warnings.catch_warnings(record=True) as w:
            colocate_fun[db_type](port=6780, ifname="lo", **db_args)
            if db_type == "deprecated":
                assert len(w) == 1
                assert issubclass(w[-1].category, DeprecationWarning)
                assert "Please use `colocate_db_tcp` or `colocate_db_uds`" in str(
                    w[-1].message
                )
    elif db_type == "uds":
        colo_model.colocate_db_uds(**db_args)

    # assert model will launch with colocated db
    assert colo_model.colocated

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

    # test restarting the colocated model

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_launch_colocated_pinned_model(fileutils, wlmutils):
    """Test the launch of a model with a colocated database and limit the
    cpus of the user application.

    This test will skip if the ``taskset`` command is not found
    """

    launcher = wlmutils.get_test_launcher()
    exp_name = "test-launch-colocated-pinned-model"
    exp = Experiment(exp_name, launcher=launcher)

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("send_data_local_smartredis.py")
    network_interface = wlmutils.get_test_interface()

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)
    colo_settings.set_tasks_per_node(1)

    colo_model = exp.create_model("colocated_pinned_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=6780, db_cpus=2, limit_app_cpus=True, debug=True, ifname=network_interface
    )

    # assert model will launch with colocated db
    assert colo_model.colocated

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
