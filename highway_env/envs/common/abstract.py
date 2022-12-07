import copy
import os
from typing import List, Tuple, Optional, Callable
import gym
from gym import Wrapper
from gym.wrappers import RecordVideo
from gym.utils import seeding
import numpy as np

from highway_env import utils
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray

#=================================================================

# code zwischen den strichen ist dem originalcode hinzugefuegt

#=================================================================

class AbstractEnv(gym.Env):
    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    
    observation_type: ObservationType
    action_type: ActionType
    _record_video_wrapper: Optional[RecordVideo]
    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }
    
    PERCEPTION_DISTANCE = 5.0 * Vehicle.MAX_SPEED
    """The maximum distance of any vehicle present in the observation [m]"""

    def __init__(self, config: dict = None) -> None:
        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        self.off_road_counter = 0

    #=================================================================
        # Safety Index
        self.phi = None
        self.sis_info = dict()
        self.set_sis_paras(sigma=0.3, k=1, n=1) # Initialwerte wie bei SIS-Paper
        self.eta = 0.01
    #=================================================================

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.reset() # wichtige Funktion!!!

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config_abstract(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "simulation_frequency": 10,  # [Hz]; muss ganzzahliges Vielfaches der policy_frequency sein damit _simulate() dies sinnvoll umsetzt
            "policy_frequency": 10,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False
        }

    #===========================SIS===================================
    def set_sis_paras(self, sigma, k, n):
        """safety Index Parameter setzen"""
        self.sis_para_sigma = sigma
        self.sis_para_k = k
        self.sis_para_n = n

    def set_slack_var(self, eta):
        self.eta = eta

    # safety index: phi(s)
    # falls Anpassung der Funktion erfolgt muss diese ebenfalls in RL-Algorithmus erfolgen
    def safety_index(self, d_min, d, dotd):
        return (self.sis_para_sigma + d_min)**self.sis_para_n - np.abs(d)**self.sis_para_n * np.sign(d) - self.sis_para_k*dotd

    def adaptive_safety_index(self):
        """
        Safety Index berechnen:
        (safety index ist aktuell auf Umgebung highway_env ausgelegt)
        Der Safety Index betrachtet andere Fahrzeuge und Fahrbahnrand mit kleinstem lateralem Abstand zu Fahrzeugmitte
        """
        # initialize safety index
        phi = -1e8
        sis_info_t = self.sis_info.get('sis_data', []) # sis_info zum Zeitpunkt t; leer zum Zeitpunkt t=0
        sis_info_tp1 = [] # sis_info zum Zeitpunkt t+1
        # counter for vehicle index
        cnt = 0 # Erklaerung: cnt = 1 -> vehicle mit Index 1 in Liste self.road.vehicles (erstes nicht ego-vehicle Fahrzeug) 

        # get data of the ego-vehicle
        ego_pos = self.vehicle.position # postion of ego-vehicle
        ego_vel = self.vehicle.velocity # velocity of ego-vehicle: [v_x, v_y]
        ego_lane_index = self.vehicle.lane_index # lane_index of ego-vehicle (_from, _to, _id)
        ego_lane_index_id = ego_lane_index[2] # lane id of the lane where ego-vehicle is driving
        ego_width = self.vehicle.WIDTH
        ego_length = self.vehicle.LENGTH

        # ego-vehicle Hilfsgroessen fuer d_min
        r = np.sqrt(ego_width**2 + ego_length**2) / 2 # Laenge von Fahrzeugmitte bis Ecke
        theta_0 = np.arcsin((ego_width/2) / r) # Winkel zwischen Fahrzeuglaengsachse und Ecke
        theta = abs(self.vehicle.heading - self.vehicle.lane.heading_at(self.vehicle.lane_offset[0])) # zusaetzlicher Drehwinkel des Fahrzeugs abzueglich der Strassenkruemmung

        for vehicle in self.road.vehicles[1:]: # self.road.vehicles ist Liste mit allen erzeugten Fahzeugen; erster Listeneintrag ist ego-vehicle
            """
            Safety Index zu anderen Fahrzeugen berechnen:
            iterate over the vehicles to compute the maximum safety index and give back phi
            and the vehicle index of highest phi
            """
            cnt += 1

            # get data of other vehicle
            veh_pos = vehicle.position # position of other vehicle

            # d = distance
            ego_to_vehicle_direction = (veh_pos - ego_pos)
            ego_to_vehicle_distance = np.linalg.norm(ego_to_vehicle_direction) # distance from ego-vehicle to vehicle
            d = ego_to_vehicle_distance # parameter d from Safety Index; always positive

            # dot d = velocity
            # berechnen, mit welcher Geschwindigkeit ego-vehicle auf anderes vehicle zufaehrt, wenn vehicle als fester Punkt betrachtet wird
            # Skalarprodukt (np.dot) berechnet den Anteil der Geschwindigkeit der auf anderes vehicle gerichtet ist 
            # und Division normiert diesen Anteil auf den Abstand zum anderen vehicle
            dotd = -np.dot(ego_vel, ego_to_vehicle_direction) / ego_to_vehicle_distance # TODO: evtl v von anderem vehicle in Berechnung von dotd einbeziehen
            # if dotd <0, then we are getting closer to hazard

            # Mindestabstand d_min zwischen Fahrzeugen: abhaengig davon ob Fahrzeuge auf derselben Spur oder nicht und von Gierwinkel des ego-vehicles
            if  ego_lane_index == vehicle.lane_index: # gilt nur wenn Fahrzeuge auf selber lane (nicht gegeben wenn ego-vehicle z.b. in Kreisverkehr faehrt)
                # Abstand aufgrund von Fahrzeuglaenge
                d_min = ego_length
            else:
                # Abstand den ego-vehicle aufgrund von Gieren braucht + halbe Breite des anderen Fahrzeugs
                d_min = (np.abs(np.sin(theta_0 + theta)) * r + ego_width/2)

            assert d_min>=0, "Error: d_min ist negativ (d_min = {})".format(d_min) # d_min muss positiv sein, sonst Fehler bei Gradientenberechnung der SI-Parameter in loss_si.backward()

            sis_info_tp1.append((d, dotd, d_min))

            # compute the safety index for specific vehicle
            phi_tmp = self.safety_index(d_min, d, dotd)
            ''' phi = sigma + d_min^n - d^n - k*dotd '''

            # select the largest safety index
            if phi_tmp > phi:
                phi = phi_tmp
                index = cnt

        """
        Safety Index zu linker und rechter Road Grenze berechnen
        """
        # mindest Abstand abhaengig von Gierwinkel berechnen (bei Gieren sind Fahrzeugecken naeher an Fahrbahnrand)
        d_min = np.abs(np.sin(theta_0 + theta)) * r # falls theta=0 dann Abstand nur ego_width/2
        # Safety Index zu linker Road Grenze
        d_left = self.vehicle.lane.DEFAULT_WIDTH/2 + self.vehicle.lane_offset[1] + ego_lane_index_id*self.vehicle.lane.DEFAULT_WIDTH # parameter d from Safety Index; positve when on road
        if d_left >= 0: # vehicle auf road
            ego_to_rb_direction_left = np.array([0, -d_left]) # TODO: funtioniert so nur auf gerader/horizontaler Strecke
            dotd_left = -np.dot(ego_vel, ego_to_rb_direction_left) / max(np.linalg.norm(d_left), 0.0001) # max() um Division durch Null zu verhindern (falls vehicle auf Begrenzungslinie)
        else: # vehicle nicht mehr auf road (Geschwindigkeit wird nicht mehr in SI Berechnung aufgenommen)
            dotd_left = 0 # kuenstlich auf 0 gesetzt; beschreibt nicht die Realitaet  
        sis_info_tp1.append((d_left, dotd_left, d_min))
        phi_tmp_left = self.safety_index(d_min, d_left, dotd_left)
        # Safety Index zu rechter Road Grenze
        d_right = self.vehicle.lane.DEFAULT_WIDTH/2 - self.vehicle.lane_offset[1] + (self.vehicle.num_lanes-1 - ego_lane_index_id)*self.vehicle.lane.DEFAULT_WIDTH
        if d_right >= 0: # vehicle auf road
            ego_to_rb_direction_right = np.array([0, d_right]) # TODO: funtioniert so nur auf gerader/horizontaler Strecke  
            dotd_right = -np.dot(ego_vel, ego_to_rb_direction_right) / max(np.linalg.norm(d_right), 0.0001) # max() um Division durch Null zu verhindern (falls vehicle auf Begrenzungslinie)
        else: # vehicle nicht mehr auf road (Geschwindigkeit wird nicht mehr in SI Berechnung aufgenommen)
            dotd_right = 0
        sis_info_tp1.append((d_right, dotd_right, d_min))
        phi_tmp_right = self.safety_index(d_min, d_right, dotd_right)

        phi_tmp = max(phi_tmp_left, phi_tmp_right)
        # pruefen ob Safety Index zu Road Grenzen groesser ist als groesster SI zu anderen Fahrzeugen
        if phi_tmp > phi:
            cnt += 1
            phi = phi_tmp
            index = cnt

        self.sis_info.update(dict(sis_data=sis_info_tp1, sis_trans=(sis_info_t, sis_info_tp1)))

        return phi, index        
        
    #===========================SIS===================================

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = self.config["simulation_frequency"] \
            if self._record_video_wrapper else self.config["policy_frequency"]
        self.metadata['video.frames_per_second'] = video_real_time_ratio * frames_freq

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"]) # zb observation_typ = class KinematicObservation; observation dict an Observation Klasse weitergegeben
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        """
        whether a `terminal state` (as defined under the MDP of the task) is reached.
        In this case further step() calls could return undefined results.

        :return: is the state terminal
        """
        raise NotImplementedError

    def _is_truncation(self) -> bool:
        """
        whether a truncation condition outside the scope of the MDP is satisfied.
        Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
        Can be used to end the episode prematurely before a `terminal state` is reached.

        :return: is the state truncated
        """
        raise NotImplementedError

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
 	
        # info dict
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }

        #=================================================================
        # sis
        old_phi = self.phi
        self.phi, veh_index = self.adaptive_safety_index()
        # cost = phi(s') - max{phi(s)-eta, 0}
        self.delta_phi = self.phi - max(old_phi-self.eta, 0)
        self.delta_phi = np.clip(self.delta_phi, -0.1, 100)   # negative Begrenzung, damit policy nicht versucht möglichst großen Abstand zu vorfahrenden Fahrzeugen zu halten; wird aber auch in Buffer gemacht
        # update info dict
        info.update({'delta_phi': self.delta_phi})
        info.update({'phi': self.phi})
        info.update(self.sis_info)
        info.update({'veh_index': veh_index}) # vehicle index fuer Fahrzeug mit groesstem Safety Index bzw. Road Grenze
        #=================================================================

        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        return info

    def _cost(self, action: Action) -> float:
        """
        A constraint metric, for budgeted MDP.

        If a constraint is defined, it must be used with an alternate reward that doesn't contain it as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def reset(self) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self.off_road_counter = 0

        self._reset() # in subclass definiert: hier werden road und vehicles gesetzt
        self.define_spaces() # Second, to link the obs and actions to the vehicles once the scene is created

        # sis
        self.phi = self.adaptive_safety_index()[0] # nach self._reset() einbinden

        return self.observation_type.observe()

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        truncation = self._is_truncation()
        info = self._info(obs, action)

        return obs, reward, terminal, truncation, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """
        Perform several steps of simulation with constant action.
        road.step() in frequency of "simulation_frequency"; new action in frequency of "policy_frequency"
        -> in every step the constant action is simulated until a new action comes in 
        """
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle if
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"]) # step(dt)
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self) -> List[int]:
        return self.action_type.get_available_actions()

    def set_record_video_wrapper(self, wrapper: RecordVideo):
        self._record_video_wrapper = wrapper
        self.update_metadata()

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.
        If a RecordVideo wrapper has been set, use it to capture intermediate frames.
        """
        if self.viewer is not None and self.enable_auto_render:

            if self._record_video_wrapper and self._record_video_wrapper.video_recorder:
                self._record_video_wrapper.video_recorder.capture_frame()
            else:
                self.render(self.rendering_mode)

    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy

    def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
        """
        Change the type of all vehicles on the road

        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behavior(self) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', '_record_video_wrapper']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result




class MultiAgentWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = info["agents_rewards"]
        done = info["agents_dones"]
        return obs, reward, done, info