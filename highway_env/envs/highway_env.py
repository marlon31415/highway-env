import numpy as np
from typing import Optional
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray

#==================================================#
#                   HighwayEnv                     #
#==================================================#
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
                "type": "Kinematics",           # types aus 'highway_env\envs\common\observation'
                "features": ["x", "y", "vx", "vy"], # features die in Observation auftauchen sollen
                "vehicles_count":     7,        # Number of observed vehicles (incl. ego-vehicle) -> 7 fuer neue observation!
                "observe_intentions": False,    # False = standard
                "absolute":           True,     # False = Koordinaten im observation_space sind relativ zum ego-vehicle; ego-vehicle KO bleiben absolut
                "normalize":          False,    # normalsiert Observation zwischen -1 und 1; True = standard
                "add_indiv_ego_obs":  True,     # definiert ob an observation noch zusaetzliche Reihe mit observations nur fuer ego-vehicle angehaengt werden soll
            },
            "action": {
                "type": "ContinuousAction",
            },
            "lanes_count":           3,       # Anzahl Spuren
            "vehicles_count":        15,      # Anzahl Fahrzeuge, die auf der Road erzeugt werden (ohne ego-vehicle)
            "controlled_vehicles":   1,       # Anzahl der zu steuernden vehicles (1 ist standard)
            "initial_lane_id":       None,    # wenn None: zufaellige initiale Spur fuer zu steuerndes vehicle
            "duration":              40,      # [s]
            "ego_spacing":           1,       # mind. Abstand zu ego-vehicle / ratio of spacing to the front vehicle: 12+1.0*speed * spacing
            "vehicles_density":      1.5,       # >1 verringert spacing beim Platzieren der vehicles
            "collision_reward":      0,       # default=-1 ; The reward received when colliding with a vehicle.
            "right_lane_reward":     0,     # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
            "high_speed_reward":     1,     # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
            "lane_change_reward":    0,       # The reward received at each lane change action.
            "middle_of_lane_reward": 0.,     # The reward received when driving in the middle of the lane (abs(vehicle.lane_offset[1]) < value)
            "reward_speed_range":    [20, 30],# [m/s] nur in diesem Bereich gibt es reward fuer Geschwindigkeit
            "collision_trunc":       True,    # definiert ob Durchlauf mit crash des Fahrzeugs endet; default: True
            "offroad_trunc":         True,    # definiert ob Durchlauf mit Verlassen des Fahrzeugs von der Strasse endet; default: False
            "speed_trunc":           True,
            "speed_limit":           30,      # v_max auf Road
            "prediction_type":       "zero_steering", # soll Trajektorie mit konstanter Geschwindigkeit und "constant_steering" oder "zero_steering" berechnet werden
            "reward_mode":           "learn"  # "learn": for RL and "performance": to compare different policies      
        })
        return config


    def _reset(self) -> None:
        """erzeugt die Road auf der Autos platziert werden und die Autos selbst"""
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes without vehicles"""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=self.config["speed_limit"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road"""
        # definieren von welchen Typ die anderen Fahrzeuge sein sollen
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        # Wird Liste der Laenge config["controlled_vehicles"] mit einem ego-vehicle in jedem Eintrag.
        # Enthaelt nur ein ego-vehicle wenn config["controlled_vehicles"]=1 
        self.controlled_vehicles = []
        
        others = self.config["vehicles_count"]
        others_before_ego = int(np.round(np.random.uniform(1/4,1/3)*others))
        others_after_ego  = others - others_before_ego

        for _ in range(others_before_ego):
                """ Autos erzeugen, die hinter ego-vehicle fahren """
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], speed=np.random.uniform(20,25))
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        
        # ego-vehicle erzeugen
        vehicle = Vehicle.create_random(
                self.road,
                speed=self.road.np_random.uniform(20,30), # wenn None dann wird v abhaengig von speed_limit oder zufaellig in Intervall [Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]] gewaehlt
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"] / self.config["vehicles_density"]
            )
        # vehicle class wird mit vorher generierten Daten aufgerufen -> es wird vehicle erzeugt
        vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed, self.config["prediction_type"])
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.insert(0, vehicle) # vehicles-liste in Road-klasse
        
        for _ in range(others_after_ego):
                """ Autos erzeugen, die vor ego-vehicle fahrren """
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], speed=np.random.uniform(17,22))
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)        
        
        # # Funktion wichtig, wenn config["controlled_vehicles"] > 1 -> Liste mit so vielen Eintraegen wie num_bins und jeder Eintrag ca. gleich groß, z.B. [3,3,3] für 3 controlled vehicles mit 9 anderen Fahrzeugen
        # # wenn num_bins gegeben (und =1 =nur ein "controlled_vehicle") dann ist Ergebnis von near_split eine Liste = ["vehicles_count"]
        # other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        # # for-for-Schleife erzeugt fuer jedes ego-vehicle eine entsprechende Anzahl an fremden Verkehrsteilnehmern (ca. gleich aufgeteilt)
        # for others in other_per_controlled:
        #     """erzeugt Liste mit ego-vehicles und setzt diese auf die Road"""
        #     # zufaellige Daten zum erzeugen des vehicles
        #     vehicle = Vehicle.create_random(
        #         self.road,
        #         speed=self.road.np_random.uniform(20,30), # wenn None dann wird v abhaengig von speed_limit oder zufaellig in Intervall [Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]] gewaehlt
        #         lane_id=self.config["initial_lane_id"],
        #         spacing=self.config["ego_spacing"]
        #     )
        #     # vehicle class wird mit vorher generierten Daten aufgerufen -> es wird vehicle erzeugt
        #     vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed, self.config["prediction_type"])
        #     self.controlled_vehicles.append(vehicle)
        #     self.road.vehicles.append(vehicle) # vehicles-liste in Road-klasse

        #     for _ in range(others):
        #         """erzeugt andere Verkehrsteilnehmer mit Methode create_random() von class Vehicle(RoadObject)"""
        #         if _ == 0:
        #             # erstes Auto soll großen Abstand zu ego-vehicle haben, damit Initialtrajektorie feasible!
        #             vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"]) # spacing = 6
        #         else:
        #             vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
        #         vehicle.randomize_behavior()
        #         self.road.vehicles.append(vehicle) # self.road.vehicles[1:] sind fremde Autos (alle ab Index 1)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # Liste aus allen Spuren auf der Road, die dieselben Indexe _from und _to haben: 
        # [(_from, _to, _id: 0), (_from, _to, _id: 1), (_from, _to, _id: 2), ...]
        num_lanes = self.vehicle.num_lanes
        # lane id der Spur des ego-vehicles; rechte Spur hat _id = config["lanes_count"]-1
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        lane_width = self.vehicle.lane.DEFAULT_WIDTH
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.velocity[0] # berechnet mit Gierwinkel psi und Schwimmwinkel beta
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        
        """reward berechnen: aehnlich wie in MPC"""
        speed_goal = 25
        lane_goal  = 8
        reward_scale = 40
        
        if self.config["reward_mode"] == "learn":
            # reward for RL
            reward = \
                - self.config["right_lane_reward"] * (self.vehicle.target_lane_offset[1])**2 \
                - self.config["high_speed_reward"] * (speed_goal - forward_speed)**2
            
            reward /= reward_scale # skalieren damit reward nicht zu groß
            offset = ( self.config["right_lane_reward"] * (lane_goal+2)**2 + self.config["high_speed_reward"] * speed_goal**2 ) / reward_scale # maximaler negativer reward
            reward += offset                                             # reward von [-offset,0] auf [0,offset] transformieren
            reward = np.exp(reward) - 1                                  # reward exponentiell werten (reward=0 ergibt wieder 0 durch die -1)
            reward = utils.lmap(reward, [0, (np.exp(offset)-1)], [0, 1]) # reward auf Intervall [0,1] normalisieren 
            reward += self.config["collision_reward"] * self.vehicle.crashed
        else:
            # reward to compare different policies (e.g. MPC and RL)
            reward = - 0.1*self.vehicle.target_lane_offset[1]**2 - 0.9*(speed_goal - forward_speed)**2

        """alternativer reward: wie urspruenglich vorgeschlagen"""
        ## collision reward: bei crash mit anderem Fahrzeug \
        ## + right lane reward: f(x) = 1 / (num_lanes-1)^2 * lane^2 -> diskrete Funktion; auf lane 0 immer Null und auf rechter lane (num_lanes-1) immer 1; dazwischen quadratisch \
        ## + high speed reward: nur zwischen definiertem Bereich gibt es reward \
        ## + middle of lane reward: reward nur wenn ego-vehicle bestimmten lateralen Abstand zur Spurmitte hat: kontinuierliche quadratische Funktion f(x)=16*(x-0.25)^2 -> f(0)=1, f(0.25)=0  
        ## Abstand zur Mitte bis zu dem es noch reward gibt
        # lat_reward = 0.25  # wenn lat_reward veraendert wird muss Vorfaktor veraendert werden!!!
        # reward = \
            # + self.config["collision_reward"]      * self.vehicle.crashed \
            # + self.config["right_lane_reward"]     * 1/max(num_lanes - 1, 1)**2 * lane**2\
            # + self.config["high_speed_reward"]     * np.clip(scaled_speed, 0, 1) \
            # + self.config["middle_of_lane_reward"] * (1-abs(self.vehicle.lane_offset[1])/(lane_width/2))
            # + self.config["middle_of_lane_reward"] * np.clip(16*(np.clip(abs(self.vehicle.lane_offset[1]), 0, lat_reward)-lat_reward)**2, 0, 1) 
        # reward = utils.lmap(reward, [self.config["collision_reward"], self.config["right_lane_reward"]+self.config["high_speed_reward"]+self.config["middle_of_lane_reward"]], [0, 1]) 
        # reward = 0 if forward_speed < self.config["reward_speed_range"][0] else reward
        # reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """
        The episode is over when ego-vehicle crashes or drives offroad
        TODO: define a goal -> terminates environment too
        """
        on_road = self.vehicle.on_road
        # Counter um aufeinanderfolgende Offroad Schritte zu zaehlen 
        if on_road == False:
            self.off_road_counter += 1 
        else:
            self.off_road_counter = 0

        return (self.config["collision_trunc"] and self.vehicle.crashed) or \
               (self.config["offroad_trunc"]   and self.off_road_counter >= 3) or \
               (self.config["speed_trunc"]     and self.vehicle.speed < 16.5) 

    def _is_truncation(self) -> bool:
        """
        whether a truncation condition outside the scope of the MDP is satisfied.
        Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
        Can be used to end the episode prematurely before a `terminal state` is reached.
        """
        return self.time >= self.config['duration']
            

    def _cost(self, action: int) -> float:
        """ 
        cost ist 1, wenn Bedingung [phi(s') - max{phi(s)-eta, 0} <= 0] verletzt wird 
        """
        return int(self.delta_phi > 0)
        """
        "The cost signal is the occurrence of unsafe states (collision and offroad)."
        cost = {}
        cost['cost_crash'] = 0
        cost['cost_road_boundary'] = 0
    
        if self.vehicle.crashed:
            cost['cost_crash'] = 1 # TODO: ueber alle vehicles interieren, falls mehrere crashs vorliegen (_is_colliding(vehicle[i+1], ego-vehicle) und intersecting pruefen
        if not self.vehicle.on_road:
            cost['cost_road_boundary'] = 1

        sum_cost = 0
        for k in list(cost.keys()):
            # cost summieren
            sum_cost += cost[k]
        if sum_cost >= 1:
            # cost kann maximal 1 sein
            sum_cost = 1

        return sum_cost
        """

     


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



#==================================================#
#                   register env                   #
#==================================================#
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