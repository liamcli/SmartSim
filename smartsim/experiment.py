import pickle
import sys
import zmq

from os import path, mkdir, listdir, getcwd, environ
from .error import SmartSimError, SSConfigError
from .orchestrator import Orchestrator
from .entity import SmartSimEntity, SmartSimNode, NumModel, Ensemble
from .generation import Generator
from .control import Controller
from .launcher import LocalLauncher

from .utils import get_logger
logger = get_logger(__name__)


class Experiment:
    """In SmartSim, the Experiment class is an entity creation API
       that both houses and operates on the entities it creates.
       The entities that can be created are:

         - NumModels
         - Ensembles
         - SmartSimNodes
         - Orchestrator

        Each entity has a distinct purpose within an experiment.

        NumModel
        --------
        Instances of numerical models or "simulation" models. NumModels can
        be created through a call to Experiment.create_model() and though
        the creation of an Ensemble.

        Ensemble
        --------
        Ensembles are groups of NumModels to be either generated manually or
        through a call to experiment.generate(). Ensembles can be given model
        parameters to be written into input files for the model at runtime.
        There are multiple ways of generating ensemble members; see
        experiment.generate() for details.

        SmartSimNodes
        -------------
        Nodes run processes adjacent to the simulation. Nodes can be used
        for anything from analysis, training, inference, etc. Nodes are the
        most flexible entity with no requirements on language or framework.
        Nodes are commonly used for acting on data being streamed out of a
        simulation model through the orchestrator

        Orchestrator
        ------------
        The Orchestrator is a KeyDB database, clustered or standalone, that
        is launched prior to the simulation. The Orchestrator can be used
        to store data from another entity in memory during the course of
        an experiment. In order to stream data into the orchestrator or
        recieve data from the orchestrator, one of the SmartSim clients
        has to be used within a NumModel or SmartSimNode. Use
        experiment.register_connection() to connect two entities within
        SmartSim.
    """
    def __init__(self, name, launcher="slurm"):
        """Initialize an Experiment

        :param name: Name of the experiment
        :type name: str
        :param launcher: type of launcher, options are "local" and "slurm",
                         defaults to "slurm"
        :type launcher: str, optional
        """
        self.name = name
        self.ensembles = []
        self.nodes = []
        self.orc = None
        self.exp_path = path.join(getcwd(), name)
        self._control = Controller(launcher=launcher)

    def start(self, ensembles=None, ssnodes=None, orchestrator=None):
        """Start the experiment by turning all entities into jobs
           for the underlying launcher specified at experiment
           initialization. All entities in the experiemnt will be
           launched if not arguments are passed.

        :param ensembles: list of Ensembles, defaults to None
        :type ensembles: list Ensemble instances, optional
        :param ssnodes: list of SmartSimNodes, defaults to None
        :type ssnodes: list of SmartSimNode instances, optional
        :param orchestrator: Orchestrator instance, defaults to None
        :type orchestrator: Orchestrator, optional
        """
        logger.info(f"Starting experiment: {self.name}")
        try:
            if not ensembles:
                ensembles = self.ensembles
            if not ssnodes:
                ssnodes = self.nodes
            if not orchestrator:
                orchestrator = self.orc

            self._control.start(
                ensembles=ensembles,
                nodes=ssnodes,
                orchestrator=orchestrator)
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop(self, ensembles=None, models=None, nodes=None, orchestrator=None):
        """Stop specific entities launched through SmartSim. This method is only
           applicable when launching with a workload manager.

           :param ensembles: Ensemble objects to be stopped
           :type Ensembles: list of Ensemble objects
           :param models: Specific models to be stopped
           :type models: list of Model objects
           :param nodes: SmartSimNodes to be stopped
           :type nodes: list of SmartSimNodes
           :param orchestrator: the orchestrator to be stoppped
           :type orchestrator: instance of Orchestrator
           :raises: SmartSimError
        """
        try:
            self._control.stop(
                ensembles=ensembles,
                models=models,
                nodes=nodes,
                orchestrator=orchestrator
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_allocation(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        """Get an allocation through the launcher for future calls
           to start to launch entities onto. This allocation is
           maintained within SmartSim. To release the allocation
           call Experiment.release().

           The kwargs can be used to pass extra settings to the
           workload manager such as the following for Slurm:
             - nodelist="nid00004"

           For arguments without a value, pass None or and empty
           string as the value for the kwarg. For Slurm:
             - exclusive=None

        :param nodes: number of nodes for the allocation
        :type nodes: int
        :param ppn: processes per node
        :type ppn: int
        :param duration: length of the allocation in HH:MM:SS format,
                         defaults to "1:00:00"
        :type duration: str, optional
        :raises SmartSimError: if allocation could not be obtained
        :return: allocation id
        :rtype: str
        """
        try:
            alloc_id = self._control.get_allocation(
                nodes=nodes,
                ppn=ppn,
                duration=duration,
                **kwargs
            )
            return alloc_id
        except SmartSimError as e:
            logger.error(e)
            raise e

    def add_allocation(self, alloc_id):
        """Add an allocation to SmartSim such that entities can
           be launched on it.

        :param alloc_id: id of the allocation from the workload manager
        :type alloc_id: str
        :raises SmartSimError: If the allocation cannot be found
        """
        try:
            self._control.add_allocation(alloc_id)
        except SmartSimError as e:
            logger.error(e)
            raise e

    def stop_all(self):
        """Stop all entities that were created with this experiment

            :raises: SmartSimError
        """
        try:
            self._control.stop(
                ensembles=self.ensembles,
                nodes=self.nodes,
                orchestrator=self.orc
                )
        except SmartSimError as e:
            logger.error(e)
            raise

    def release(self, alloc_id=None):
        """Release the allocation(s) stopping all jobs that are
           currently running and freeing up resources. If an
           allocation ID is provided, only stop that allocation
           and remove it from SmartSim.

        :param alloc_id: id of the allocation, defaults to None
        :type alloc_id: str, optional
        :raises SmartSimError: if fail to release allocation
        """
        try:
            self._control.release(alloc_id=alloc_id)
        except SmartSimError as e:
            logger.error(e)
            raise

    def poll(self, interval=10, poll_db=False, verbose=True):
        """Poll the running simulations and recieve logging output
           with the status of the job. If polling the database,
           jobs will continue until database is manually shutdown.

           :param int interval: number of seconds to wait before polling again
           :param bool poll_db: poll dbnodes for status as well and see
                                it in the logging output
           :param bool verbose: set verbosity
           :raises: SmartSimError
        """
        try:
            self._control.poll(interval, poll_db, verbose)
        except SmartSimError as e:
            logger.error(e)
            raise


    def finished(self, entity):
        """Return a boolean indicating wether or not a job has finished.

           :param entity: object launched by SmartSim. One of the following:
                          (SmartSimNode, NumModel, Orchestrator, Ensemble)
           :type entity: SmartSimEntity
           :returns: bool
        """
        try:
            return self._control.finished(entity)
        except SmartSimError as e:
            logger.error(e)
            raise


    def generate(self, model_files=None, node_files=None, tag=None,
                 strategy="all_perm", **kwargs):
        """Generate the file structure for a SmartSim experiment. This
           includes the writing and configuring of input files for a
           model. Ensembles created with a 'params' argument will be
           expanded into multiple models based on a generation strategy.
           Model input files are specified with the model_files argument.
           All files and directories listed as strings in a list will be
           copied to each model within an ensemble. Every model file is read,
           checked for input variables to configure, and written. Input
           variables to configure are specified with a tag within the input
           file itself. The default tag is surronding an input value with
           semicolons. e.g. THERMO=;90;

           Files for SmartSimNodes can also be included to be copied into
           node directories but are never read nor written. All node_files
           will be copied into directories named after the name of the
           SmartSimNode within the experiment.

            :param model_files: The model files for the experiment.  Optional
                                if model files are not needed for execution.
            :type model_files: list of path like strings to directories or files
            :param node_files: files to be copied into node directories. These
                               are most likely files needed to run the node
                               computations. e.g. a python script
            :type node_files: list of path like strings to directories or files

            :param str strategy: The permutation strategy for generating models within
                                ensembles.
                                Options are "all_perm", "random", "step", or a
                                callable function. Defaults to "all_perm"
            :raises SmartSimError: if generation fails
        """
        try:
            generator = Generator()
            generator.set_strategy(strategy)
            if tag:
                generator.set_tag(tag)
            generator.generate_experiment(
                self.exp_path,
                ensembles=self.ensembles,
                nodes=self.nodes,
                orchestrator=self.orc,
                model_files=model_files,
                node_files=node_files,
                **kwargs
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_status(self, entity):
        """Get the status of a running job that was launched through
           a workload manager. Ensembles, Orchestrator, SmartSimNodes,
           and NumModel instances can all be passed to have their
           status returned as a string. The type of string and content
           will depend on the workload manager being used.

           :param entity: The SmartSimEntity object that was launched
                          to check the status of
           :type entity: SmartSimEntity
           :returns: status of the entity
           :rtype: list if entity contains sub-entities such as cluster
                   Orchestrator or Ensemble
           :raises SmartSimError: if status retrieval fails
           :raises TypeError: if one argument was not a SmartSimEntitiy
        """
        try:
            if isinstance(entity, Ensemble):
                return self._control.get_ensemble_status(entity)
            elif isinstance(entity, Orchestrator):
                return self._control.get_orchestrator_status(entity)
            elif isinstance(entity, NumModel):
                return self._control.get_model_status(entity)
            elif isinstance(entity, SmartSimNode):
                return self._control.get_node_status(entity)
            else:
                raise TypeError(
                    f"entity argument was of type {type(entity)} not SmartSimEntity")
        except SmartSimError as e:
            logger.error(e)
            raise


    def create_ensemble(self, name, params={}, run_settings={}):
        """Create a ensemble to be used within one or many of the
           SmartSim Modules. Ensembles contain groups of models.
           Parameters can be given to a ensemble in order to generate
           models based on a combination of parameters and generation
           strategies. For more on generation strategies, see the
           Generator Class.

        :param name: name of the ensemble
        :type name: str
        :param params: model parameters for generation strategies,
                       defaults to {}
        :type params: dict, optional
        :param run_settings: define how the model should be run,
                             defaults to {}
        :type run_settings: dict, optional
        :raises SmartSimError: If ensemble cannot be created
        :return: the created Ensemble
        :rtype: Ensemble
        """
        try:
            new_ensemble = None
            for ensemble in self.ensembles:
                if ensemble.name == name:
                    raise SmartSimError("A ensemble named " + ensemble.name +
                                        " already exists!")

            ensemble_path = path.join(self.exp_path, name)
            if path.isdir(ensemble_path):
                error = " ".join((
                    "ensemble directory already exists:",ensemble_path))
                raise SmartSimError(error)
            new_ensemble = Ensemble(name,
                                    params,
                                    self.name,
                                    ensemble_path,
                                    run_settings=run_settings)
            self.ensembles.append(new_ensemble)
            return new_ensemble
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_model(self, name, ensemble="default", params={}, path=None,
                     run_settings={}):
        """Create a model belonging to a specific ensemble. This function is
           useful for running a small number of models where the model files
           are already in place for execution.

           Calls to this function without specifying the `ensemble` argument
           result in the creation/usage a ensemble named "default", the default
           argument for `ensemble`.

           Models in the default ensemble will be launched with their specific
           run_settings as defined in intialization here. Otherwise the model
           will use the run_settings defined for the Ensemble

        :param name: name of the model
        :type name: str
        :param ensemble: name of the ensemble to add the model to,
                         defaults to "default"
        :type ensemble: str, optional
        :param params: model parameters for generation strategies,
                       defaults to {}
        :type params: dict, optional
        :param path: path to where the model should be executed at runtime,
                     defaults to os.getcwd()
        :type path: str, optional
        :param run_settings: defines how the model should be run,
                             defaults to {}
        :type run_settings: dict, optional
        :raises SmartSimError: if ensemble name provided doesnt exist
        :return: the created model
        :rtype: NumModel
        """
        try:
            model_added = False
            model = NumModel(name, params, path, run_settings)
            if not path:
                path = getcwd()
            if ensemble == "default" and "default" not in [
                    ensemble.name for ensemble in self.ensembles]:

                # create empty ensemble
                self.create_ensemble(ensemble, params={}, run_settings={})
            for t in self.ensembles:
                if t.name == ensemble:
                    t.add_model(model)
                    model_added = True
            if not model_added:
                raise SmartSimError("Could not find ensemble by the name of: " +
                                    ensemble)
            return model
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_orchestrator_cluster(self, alloc, path=None, port=6379, db_nodes=3,
                                     dpn=1, **kwargs):
        """Create an in-memory database that can be used for the transfer of
           data between launched entities. For increasing throughput,
           the number of databases per node can be increased such that mulitple
           database instances are launched on a single node and share the
           memory of that node.

           Orchestrator clusters are meant to be launched accross multiple
           nodes with the minimum number of nodes being 3. Clusters of size
           2 are not supported and single instances will be launched in
           standalone mode. This function requires that an allocation id
           is passed for the orchestrator to run on. If you want to run
           an orchestrator locally, look at Experiment.create_orchestrator().

           Additional options for where and how to launch the orchestrator
           can be specified with the through the inclusion of extra kwargs
           (e.g. partition="main")


        :param alloc: id of the allocation for the orchestrator to run on
        :type alloc: str
        :param path: desired path for orchestrator output/error, defaults to cwd
        :type path: str, optional
        :param port: port orchestrator should run on, defaults to 6379
        :type port: int, optional
        :param db_nodes: number of database nodes in the cluster, defaults to 3
        :type db_nodes: int, optional
        :param dpn: number of databases per node, defaults to 1
        :type dpn: int, optional
        :raises SmartSimError: if an orchestrator already exists within the
                               experiment
        :raises SmartSimError: If experiment was initialized with local launcher
        :return: Orchestrator instance
        :rtype: Orchestrator
        """
        try:
            if self.orc:
                raise SmartSimError(
                    "Only one orchestrator can exist within a experiment.")
            if isinstance(self._control._launcher, LocalLauncher):
                error = "Clustered orchestrators are not supported when using the local launcher\n"
                error += "Use Experiment.create_orchestrator() for launching an orchestrator"
                error += "with the local launcher"
                raise SmartSimError(error)
            orcpath = getcwd()
            if path:
                orcpath = path

            self.orc = Orchestrator(orcpath,
                                    port=port,
                                    db_nodes=db_nodes,
                                    dpn=dpn,
                                    alloc=alloc,
                                    **kwargs)
            return self.orc
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_orchestrator(self, path=None, port=6379, **kwargs):
        """Create an in-memory database to run with an experiment. Launched
           entities can communicate with the orchestrator through use
           of one of the Python, C, C++ or Fortran clients.

           With the default settings, this function can be used to create
           a local orchestrator that will run in parallel with other
           entities running serially in an experiment. For creating
           clustered orchestrators accross multiple compute nodes
           look at Experiment.create_orchestrator_cluster()

        :param path: desired path for orchestrator output/error, defaults to cwd
        :type path: str, optional
        :param port: port orchestrator should run on, defaults to 6379
        :type port: int, optional
        :raises SmartSimError: if an orchestrator already exists
        :return: Orchestrator instance created
        :rtype: Orchestrator
        """
        try:
            if self.orc:
                raise SmartSimError(
                    "Only one orchestrator can exist within a experiment.")
            orcpath = getcwd()
            if path:
                orcpath = path

            self.orc = Orchestrator(orcpath,
                                    port=port,
                                    **kwargs)
            return self.orc
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_node(self, name, script_path=None, run_settings={}):
        """Create a SmartSimNode for a specific task. Examples of SmartSimNode
           tasks include training, processing, and inference. Nodes can be used
           to run any task written in any language. The included script/executable
           for nodes often use the Client class to send and recieve data from
           the SmartSim orchestrator.

           :param str name: name of the node to be launched
           :param str script_path: path to the script or executable to be launched.
                                   (default is the current working directory of the
                                    SmartSim run script)
           :param dict run_settings: Settings for the workload manager can be set by
                                     including keyword arguments such as
                                     duration="1:00:00" or nodes=5
           :raises: SmartSimError if node exists by the same name
           :returns: SmartSimNode created
           :rtype: SmartSimNode
           """
        try:
            for node in self.nodes:
                if node.name == name:
                    raise SmartSimError("A node named " + node.name +
                                        " already exists!")
            node = SmartSimNode(name, script_path, run_settings=run_settings)
            self.nodes.append(node)
            return node
        except SmartSimError as e:
            logger.error(e)
            raise

    def register_connection(self, sender, reciever):
        """Create a runtime connection in orchestrator for data to be passed
           between two SmartSim entities. The possible types of connections
           right now are:

                Model -> Node
                Node  -> Node
                Node  -> Model

           :param str sender: name of the created entity with a Client instance
                              to send data to a registered counterpart
           :param str reciever: name of the created entity that will recieve
                                data by making calls to a Client instance.
           :raises TypeError: if arguments are not str names of entities
           :raises SmartSimError: if orchestrator has not been created
        """
        try:
            if not isinstance(sender, str) or not isinstance(reciever, str):
                raise TypeError(
                    "Arguments to register connection must either be a str")
            if not self.orc:
                raise SmartSimError("Create orchestrator to register connections")
            else:
                # TODO check if sender and reciever are registered entities
                # TODO check for illegal connection types. e.g. model to model
                self.orc.junction.register(sender, reciever)
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_connection(self, sender):
        """Create a connection between the experiment script and
           SmartSim entity. This method should be called after a
           database has been launched or an error will be raised.

           :param str sender: name of the created entity with a
                              Client instance to send data to
                              this smartsim script
           :raises SmartSimError: if orchestrator has not been created
        """
        if not self.orc:
            raise SmartSimError("Create orchestrator to register connections")
        try:
            environ["SSDB"] = self.get_db_address()[0]
        except SmartSimError as e:
            logger.error(e)
            raise
        environ["SSNAME"] = self.name
        environ["SSDATAIN"] = sender


    def delete_ensemble(self, name):
        """Delete a created ensemble from Experiment so that
           any future calls to SmartSim Modules will not include
           this ensemble.

           :param str name: name of the ensemble to be deleted
           :raises TypeError: if argument is not a str name of an ensemble
           :raises SmartSimError: if ensemble doesnt exist
        """
        try:
            if isinstance(name, SmartSimEntity):
                name = name.name
            if not isinstance(name, str):
                raise TypeError("Argument to delete_ensemble must be of type str")
            ensemble_deleted = False
            for t in self.ensembles:
                if t.name == name:
                    self.ensembles.remove(t)
                    ensemble_deleted = True
            if not ensemble_deleted:
                raise SmartSimError("Could not delete ensemble: " + name)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_model(self, model, ensemble):
        """Get a specific model from a ensemble.

           :param str model: name of the model to return
           :param str ensemble: name of the ensemble where the model is located

           :raises SmartSimError: if model is not found
           :raises TypeError: if arguments are not str names of a model
                              and/or an Ensemble
           :returns: NumModel instance
           :rtype: NumModel
        """
        try:
            if not isinstance(ensemble, str):
                raise TypeError("Ensemble argument to get_model must be of type str")
            if not isinstance(model, str):
                raise TypeError("Model argument to get_model must be of type str")

            ensemble = self.get_ensemble(ensemble)
            model = ensemble[model]
            return model
        except KeyError:
            raise SmartSimError("Model not found: " + model)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_ensemble(self, ensemble):
        """Return a specific ensemble from Experiment

           :param str ensemble: Name of the ensemble to return
           :raises SmartSimError: if ensemble is not found
           :raises TypeError: if argument is not a str name of
                              an ensemble
           :returns: ensemble instance
           :rtype: Ensemble
        """
        try:
            if not isinstance(ensemble, str):
                raise TypeError("Argument to get_ensemble must be of type str")
            for t in self.ensembles:
                if t.name == ensemble:
                    return t
            raise SmartSimError("ensemble not found: " + ensemble)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_node(self, node):
        """Return a specific node from Experiment

           :param str node: Name of the node to return
           :raises SmartSimError: if node cannot be found
           :raises TypeError: if argument is not a str name
                    of an node
           :returns: node instance
           :rtype: SmartSimNode
        """
        try:
            if not isinstance(node, str):
                raise TypeError("Argument to get_node must be of type str")
            for n in self.nodes:
                if n.name == node:
                    return n
            raise SmartSimError("Node not found: " + node)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_db_address(self):
        """Get the TCP address of the orchestrator returned by pinging the
           domain name used by the workload manager e.g. nid00004 returns
           127.0.0.1

           :raises SmartSimError: if orchestrator has not been launched
           :raises: SmartSimError: if database nodes cannot be found
           :returns: tcp address of orchestrator
           :rtype: returns a list if clustered orchestrator
        """
        if not self.orc:
            raise SmartSimError("No orchestrator has been initialized")
        addresses = []
        for dbnode in self.orc.dbnodes:
            job = self._control._jobs[dbnode.name]
            if not job.nodes:
                raise SmartSimError("Database has not been launched yet.")

            for address in job.nodes:
                for port in dbnode.ports:
                    addr = ":".join((address, str(port)))
                    addresses.append(addr)
        if len(addresses) < 1:
            raise SmartSimError("Could not find nodes Database was launched on")
        return addresses

    def __str__(self):
        experiment_str = "\n-- Experiment Summary --\n"
        if len(self.ensembles) > 0:
            experiment_str += "\n-- ensembles --"
            for ensemble in self.ensembles:
                experiment_str += str(ensemble)
        if len(self.nodes) > 0:
            experiment_str += "\n-- Nodes --"
            for node in self.nodes:
                experiment_str += str(node)
        if self.orc:
            experiment_str += str(self.orc)
        return experiment_str