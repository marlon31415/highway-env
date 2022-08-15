'''
Aenderungen:
- default_config: 
    - "action": {"type": "ContinuousAction"} (vorher: "DiscreteMetaAction")
    - "offroad_terminal": True (vorher: False)
-
-
'''

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

#==================================================#
#                   HighwayEnv                     #
#==================================================#

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config_abstract()
        config.update({
            "observation": {
                "type": "Kinematics",  # types aus 'highway_env.envs.common.observation'
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "vehicles_count": 8,   # Number of observed vehicles 
            },
            "action": {
                "type": "ContinuousAction",
            },
            "lanes_count": 4,          # Anzahl Spuren
            "vehicles_count": 20,      # Anzahl Fahrzeuge, die auf der Road erzeugt werden (ohne ego-vehicle)
            "controlled_vehicles": 1,  # Anzahl der zu steuernden vehicles (1 ist standard)
            "initial_lane_id": None,   # zufaellige initiale Spur fuer zu steuerndes vehicle
            "duration": 40,            # [s]
            "ego_spacing": 2,          # mind. Abstand zu ego-vehicle / ratio of spacing to the front vehicle: 12+1.0*speed * spacing
            "vehicles_density": 1,     # ?
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30], # nur in diesem Bereich gibt es reward fuer Geschwindigkeit
            "offroad_terminal": True,  # definiert ob Durchlauf auch mit Verlassen des Fahrzeugs von der Strasse endet; default: False
            "absolut": False,          # Koordinaten im observation_space sind relativ zum ego-vehicle; ego-vehicle KO bleiben absolut
        })
        return config


    def _reset(self) -> None:
        """erzeugt die Road auf der Autos platziert werden und die Autos selbst"""
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes without vehicles"""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road"""
        # definieren von welchen Typ die anderen Fahrzeuge sein sollen
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # Funktion wichtig, wenn config["controlled_vehicles"] > 1 -> Liste mit so vielen Eintraegen wie num_bins und jeder Eintrag ca. gleich groß.
        # wenn num_bins gegeben (und =1 =nur ein "controlled_vehicle") dann ist Ergebnis von near_split eine Liste = ["vehicles_count"]
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        # Wird Liste der Laenge config["controlled_vehicles"] mit einem ego-vehicle in jedem Eintrag.
        # Enthaelt nur ein ego-vehicle wenn config["controlled_vehicles"]=1 
        self.controlled_vehicles = []

        # for-for-Schleife erzeugt fuer jedes ego-vehicle eine entsprechende Anzahl an fremden Verkehrsteilnehmern (ca. gleich aufgeteilt)
        for others in other_per_controlled:
            """erzeugt Liste mit ego-vehicles und setzt diese auf die Road"""
            # zufaellige Daten zum erzeugen des vehicles
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            # vehicle class wird mit vorher generierten Daten aufgerufen -> es wird vehicle erzeugt
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle) # vehicles-liste in Road-klasse

            for _ in range(others):
                """erzeugt andere Verkehrsteilnehmer mit Methode create_random() von class Vehicle(RoadObject)"""
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle) # self.road.vehicles[1:] sind fremde Autos (alle ab indize 1)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1]) # normalisiert reward auf dem Intervall [0,1]
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.time >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

     


#==================================================#
#                HighwayEnv Fast                   #
#==================================================#
class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

class MOHighwayEnv(HighwayEnv):
    """
    A multi-objective version of HighwayEnv
    """

    def _rewards(self, action: Action) -> dict:
        """
        In MORL, we consider multiple rewards like collision, right-keeping, and speed,
        and the utility of these separate rewards is not always known a priori.
        This function returns a dict of multiple reward criteria

        :param action: the last action performed
        :return: the reward vector
        """
        rewards = {}

        rewards["collision"] = self.vehicle.crashed

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        rewards["right_lane"] = lane / max(len(neighbours) - 1, 1)

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        rewards["high_speed"] = np.clip(scaled_speed, 0, 1)

        return rewards

    def _reward(self, action: Action) -> float:
        """
        This scalarized reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        
        :param action: the last action performed
        :return: the reward
        """
        rewards = self._rewards(action)
        reward = \
            + self.config["collision_reward"] * rewards["collision"] \
            + self.config["right_lane_reward"] * rewards["right_lane"] \
            + self.config["high_speed_reward"] * rewards["high_speed"]
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of rewards

        :param obs: current observation
        :param action: current action
        :return: reward dict
        """
        return self._rewards(action)

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)

register(
    id='mo-highway-v0',
    entry_point='highway_env.envs:MOHighwayEnv',
)