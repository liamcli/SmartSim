import sys
import shutil

from itertools import product
from os import mkdir, getcwd, path
from distutils import dir_util

from .model import NumModel
from .modelwriter import ModelWriter
from ..error import SmartSimError, SSUnsupportedError
from ..helpers import get_SSHOME
from ..ssModule import SSModule

"""
Generation

 - models are created based on the content of the simulation.toml
   that will be populated as a result of the interface or manual
   creation.
 - models are created with the following tree for and example 1 target with
   two resulting models
   - lammps_atm/              (experiment name)
     └── atm                  (target name)
         ├── atm_ld           (model name)
         │    └── in.atm
         └── atm_ls           (model name)
              └── in.atm

A configuration file for this generation could look like the following when generated
with the all permutations strategy.

```toml
[model]
name = "lammps"
targets = ["atm"]
experiment = "lammps_atm"
configs = ["in.atm"]

[atm]
  [atm.lj]              # lj is the previous value marked in "in.atm" (e.g. ;lj;)
  value = ["ls", "ld"]
```

"""


class Generator(SSModule):
    """Data generation phase of the Smart Sim pipeline. Holds internal configuration
       data that is created during the data generation stage.

       Args
         state  (State): The state class that manages the library
    """

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.state._set_state("Data Generation")
        self._writer = ModelWriter()


###########################
### Generator Interface ###
###########################

    def generate(self):
        """Generate model runs according to the main configuration file
           Note that this only generates the necessary files and structure
           to be able to run all models in parallel, it does not actually
           run any models."""
        try:
            self.log("SmartSim State: " + self.state.get_state())
            self._create_models()
            self._create_experiment()
            self._configure_models()
        except SmartSimError as e:
            print(e)
            sys.exit()

    def set_tag(self, tag, regex=None):
        """Set the tag for the model files where configurations should
           be replaced.

           Args
              tag    (str): a string of characters that signify an string to be changed
              regex  (str): a regular expression that model files are tagged with
        """
        self._writer._set_tag(tag, regex)


##########################

    def _create_models(self):
        """Populates instances of NumModel class for all target models.
           NumModels are created via a strategy of which there is only
           one implemented: all permutations.

           This strategy takes all permutations of available configuration
           values and creates a model for each one.

           Returns: List of models with configurations to be written
        """

        # collect all parameters, names, and settings
        def read_model_parameters(target):
            target_params = target.get_target_params()
            param_names = []
            parameters = []
            for name, val in target_params.items():
                param_names.append(name)
                # if it came from a simulation.toml
                if isinstance(val, dict):
                    if isinstance(val["value"], list):
                        parameters.append(val["value"])
                    else:
                        parameters.append([val["value"]])
                # if the user called added a target programmatically
                elif isinstance(val, list):
                    parameters.append(val)
                else:
                    raise SmartSimError(self.state.get_state,
                     "Incorrect type for target parameters")
                    # TODO improve this error message
            return param_names, parameters


        # init model classes to hold parameter information
        targets = self._get_targets()
        for target in targets:
            names, values = read_model_parameters(target)
            all_configs = self._create_all_permutations(names, values)
            for i, conf in enumerate(all_configs):
                model_name = "_".join((target.name, str(i)))
                m = NumModel(model_name, conf, i)
                target.add_model(m)

    def _create_experiment(self):
        """Creates the directory structure for the simulations"""
        exp_path = self._get_exp_path()

        try:
            mkdir(exp_path)
            targets = self._get_targets()
            for target in targets:
                target_dir = path.join(exp_path, target.name)
                mkdir(target_dir)

        except FileExistsError:
            raise SmartSimError(self.state.get_state(),
                           "Data for an experiment by this name already exists!")



    def _configure_models(self):
        """Duplicate the base configurations of target models"""

        listed_configs = self._get_config(["model", "model_files"])
        exp_path = self._get_exp_path()
        targets = self._get_targets()

        for target in targets:
            target_models = target.get_models()

            # Make target model directories
            for name, model in target_models.items():
                dst = path.join(exp_path, target.name, name)
                mkdir(dst)
                model.set_path(dst)

                if not isinstance(listed_configs, list):
                    listed_configs = [listed_configs]
                for config in listed_configs:
                    dst_path = path.join(dst, path.basename(config))
                    config_path = path.join(get_SSHOME(), config)
                    if path.isdir(config_path):
                        dir_util.copy_tree(config_path, dst)
                    else:
                        shutil.copyfile(config_path, dst_path)

                # write in changes to configurations
                self._writer.write(model)



######################
### run strategies ###
######################

    # create permutations of all parameters
    # single model if parameters only have one value
    @staticmethod
    def _create_all_permutations(param_names, param_values):
        perms = list(product(*param_values))
        all_permutations = []
        for p in perms:
            temp_model = dict(zip(param_names, p))
            all_permutations.append(temp_model)
        return all_permutations

    @staticmethod
    def _one_per_change():
        raise NotImplementedError

    @staticmethod
    def _hpo():
        raise NotImplementedError

