from os import getcwd

import pytest

from smartsim import Experiment
from smartsim.entity import Ensemble, EntityList, Model
from smartsim.settings import RunSettings


def test_entityList_init():
    with pytest.raises(NotImplementedError):
        ent_list = EntityList("list", getcwd(), perm_strat="all_perm")


def test_entityList_type():
    exp = Experiment("name")
    ens_settings = RunSettings("python")
    ensemble = exp.create_ensemble("name", replicas=4, run_settings=ens_settings)
    assert ensemble.type == "Ensemble"


def test_entityList_getitem():
    """EntityList.__getitem__ is overridden in Ensemble, so we had to pass an instance of Ensemble
    to EntityList.__getitem__ in order to add test coverage to EntityList.__getitem__
    """
    exp = Experiment("name")
    ens_settings = RunSettings("python")
    ensemble = exp.create_ensemble("name", replicas=4, run_settings=ens_settings)
    assert EntityList.__getitem__(ensemble, "name_3") == ensemble[3]


def test_entityList_repr():
    exp = Experiment("name")
    ens_settings = RunSettings("python")
    ensemble = exp.create_ensemble("ens_name", replicas=1, run_settings=ens_settings)
    assert ensemble.__repr__() == "ens_name"
