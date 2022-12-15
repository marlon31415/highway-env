from typing import Union, Optional, Tuple, List
import numpy as np
import copy
from collections import deque

from highway_env import utils
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.objects import RoadObject, Obstacle, Landmark
from highway_env.utils import Vector


class Vehicle(RoadObject):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_INITIAL_SPEEDS = [23, 28] # wird beim Erzeugen von vehicle verwendet, wenn kein speed vorgegeben und speed_limit=None
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -40.
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""
    L_AXLE = 3.0
    """ wheel base (distance between front and rear axle) [m] """
    CG_FRONT = L_AXLE*0.4
    """ Distance from Center of Gravity (CG) to Front Axle [m] """
    CG_REAR = L_AXLE-CG_FRONT
    """ Distance from Center of Gravity (CG) to Rear Axle [m] """

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 prediction_type: str = 'zero_steering'):
        super().__init__(road, position, heading, speed)
        self.beta = 0 # Schwimmwinkel bei Initialisierung zu Null setzen
        self.prediction_type = prediction_type
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False # definiert ob vehicle anderes Objekt beruehrt
        self.impact = None   # definiert ob
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)

    @classmethod
    def create_random(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      spacing: float = 1) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.666*lane.speed_limit, 0.733*lane.speed_limit) # [0.666;0.833] -> [20;25]; [20;22] -> [0.666;0.733]; [15;20] -> [0.5;0.666]
            else:
                speed = road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12+1.0*speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.
        Before using action in step() it gets adjusted in clip_actions()

        :param action: the input action
        """
        if action:
            self.action = action

    #===========================================================#
    #    Implementierung des kinematischen Bicycle Modells      #
    #===========================================================#
    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions() # falls Randbedingungen verletzt wurden oder crash
        # Lenkwinkel (input)
        delta_f = self.action['steering'] # Wert ist schon im Bereich STEERING_RANGE [rad] (vgl. ContinuousAction(ActionType))
        # Schwimmwinkel
        self.beta = np.arctan(self.CG_REAR / self.L_AXLE * np.tan(delta_f))
        # Geschwindigkeit in x- und y-Richtung
        v = self.speed * np.array([np.cos(self.heading + self.beta),
                                   np.sin(self.heading + self.beta)])
        # Integration der Geschwindigkeit fuer Position
        self.position += v * dt
        # falls impact (is not None wenn Variable will_intersect von Methode _is_colliding() in Methode handle_collisions() in Klasse RoadObject() einen Wert hat)
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        # Gierwinkel
        self.heading += (self.speed / self.CG_REAR) * np.sin(self.beta)  * dt
        # absolute Geschwindigkeit
        self.speed += self.action['acceleration'] * dt # Wert ist schon im Bereich ACCELERATION_RANGE [m/s^2] (vgl. class ContinuousAction(ActionType))
        # on_state_update 
        self.on_state_update()

    def clip_actions(self) -> None:
        # if self.crashed:
        #     """definiert wie actions bei crash angepasst werden sollen: bremsen und aufhoeren zu lenken"""
        #     self.action['steering'] = 0
        #     self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < self.MIN_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MIN_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        if self.prediction_type == 'zero_steering':
            action = {'acceleration': 0.0, 'steering': 0.0}
        elif self.prediction_type == 'constant_steering':
            action = {'acceleration': 0.0, 'steering': self.action['steering']}
        else:
            raise ValueError("Unknown prediction type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        
        # threshold = 2/180*np.pi # Winkelunterschied zwischen Lane und Geschwindigkeit ab dem Spurwechsel in Praediktion der Trajektorie einfliessen soll
        # coordinates = self.lane.local_coordinates(self.position) # lokale Lane-KO -> diese koennen einfach in globale umgerechnet werden
        # route = [self.lane_index] # entweder Route (Liste mit Tuplen aus (from, to, id)) oder Liste mit aktuellem (from, to, id)
        # idx = 0 # index fuer Iteration in times-array
        # trajectory = []

        # # Bedingung fuer Spurwechsel: Geschwindigkeitsrichtung weicht von Richtung der Lane ab
        # theta1 = self.lane.heading_at(coordinates[0])
        # theta2 = np.arctan2(self.velocity[1], self.velocity[0])
        # if abs(theta1-theta2) > threshold:
        #     """Spurwechsel"""
            
        #     # Goal Lane fuer Fahrbahnwechsel (abhaengig von Ausrichtung des Autos und ob Auto rechts oder links von Spurmitte)
        #     if theta1-theta2 > 0:
        #         goal_lane = (route[0][0], route[0][1], route[0][2]-1) if coordinates[1]<=0 else route[0] # Spurwechsel oder nur in Richtung Spurmitte fahren
        #     else:
        #         goal_lane = (route[0][0], route[0][1], route[0][2]+1) if coordinates[1]>=0 else route[0] # Spurwechsel oder nur in Richtung Spurmitte fahren

        #     if theta1-theta2 > 0 and goal_lane[2] >= 0:
        #         """Spurwechsel nach links (pruefen ob Auto nach links will und ob es sich nicht auf linker Spur befindet)"""
        #         longitudinal = coordinates[0] # longitudinaler "Fortschritt" auf Lane
        #         lateral = coordinates[1] # Abweichung zu Lane Mitte
        #         # Hilfgroessen fuer Spurwechsel
        #         projection_to_other_lane = self.road.network.get_lane(goal_lane).position(longitudinal, 0) # Projektion auf Nebenspur (Mitte) zu der Fahrzeug wechseln will
        #         other_lane_direction = projection_to_other_lane - self.lane.position(longitudinal, lateral)
        #         # Geschwindigkeiten die zu Position integriert werden (nur zu Beginn ein mal berechnen, dann als konstant annehmen)
        #         theta_lane = self.lane.heading_at(longitudinal)
        #         lane_direction = np.array([np.cos(theta_lane), np.sin(theta_lane)])
        #         v_in_lane_direction = np.dot(self.velocity, lane_direction) / np.linalg.norm(lane_direction) # Geschwindigkeit in Richtung der Spur
        #         v_orthogonal = -np.dot(self.velocity, other_lane_direction) / np.linalg.norm(other_lane_direction) # Geschwindigkeit in Richtung der anderen Spur bzw. orthogonal zu v_in_lane_direction
                
        #         cur_lane_idx = route[0]
        #         for t in times: # oder while self.lane == get_lane(route[0])
                    
        #             if lateral <= -self.lane.DEFAULT_WIDTH/2:
        #                 lateral = self.lane.DEFAULT_WIDTH/2 # von lokalen lateralen KO der eigentlichen Spur zu denen der neuen Spur; ein bisschen unschoen aber sollte klappen
        #                 cur_lane_idx = goal_lane
        #             theta_lane = self.road.network.get_lane(cur_lane_idx).heading_at(longitudinal)
        #             # Integration der Geschwindigkeit zu globaler Position
        #             longitudinal += v_in_lane_direction*t
        #             lateral += v_orthogonal*t
        #             x, y = self.road.network.get_lane(cur_lane_idx).position(longitudinal, lateral)
        #             # Trajektorie
        #             trajectory.append((np.array([x,y]), theta_lane))
        #             # Index erhoehen
        #             idx += 1
        #             if np.linalg.norm(self.road.network.get_lane(goal_lane).position(longitudinal, 0)[1] - self.road.network.get_lane(cur_lane_idx).position(longitudinal, lateral)[1]) <= 0.3:
        #                 # wenn Abstand zwischen Mitte von Ziel-Lane und aktueller Position kleiner als 0.3m ist
        #                 break
        #         if idx <= len(times)-1: # prueft ob Praediktionshorizont schon erreicht wurde
        #             for t in times[idx:]:
        #                 longitudinal += self.speed*t # verwenden von skalarer Geschwindigkeit, da jetzt Mitte der Lane gefolgt wird
        #                 x, y = self.road.network.get_lane(goal_lane).position(longitudinal, 0)
        #                 theta_lane = self.road.network.get_lane(goal_lane).heading_at(longitudinal)
        #                 trajectory.append((np.array([x,y]), theta_lane))

        #     elif theta1-theta2 < 0 and goal_lane[2] <= (self.num_lanes-1):
        #         """Spurwechsel nach rechts (pruefen ob Auto nach rechts will und ob es sich nicht auf rechts Spur befindet)"""
        #         longitudinal = coordinates[0] # longitudinaler "Fortschritt" auf Lane
        #         lateral = coordinates[1] # Abweichung zu Lane Mitte
        #         # Hilfgroessen fuer Spurwechsel
        #         projection_to_other_lane = self.road.network.get_lane(goal_lane).position(longitudinal, 0) # Projektion auf Nebenspur (Mitte) zu der Fahrzeug wechseln will
        #         other_lane_direction = projection_to_other_lane - self.lane.position(longitudinal, lateral)
        #         # Geschwindigkeiten die zu Position integriert werden (nur zu Beginn ein mal berechnen, dann als konstant annehmen)
        #         theta_lane = self.lane.heading_at(longitudinal)
        #         lane_direction = np.array([np.cos(theta_lane), np.sin(theta_lane)])
        #         v_in_lane_direction = np.dot(self.velocity, lane_direction) / np.linalg.norm(lane_direction) # Geschwindigkeit in Richtung der Spur
        #         v_orthogonal = np.dot(self.velocity, other_lane_direction) / np.linalg.norm(other_lane_direction) # Geschwindigkeit in Richtung der anderen Spur bzw. orthogonal zu v_in_lane_direction
                
        #         cur_lane_idx = route[0]
        #         for t in times: # oder while self.lane == get_lane(route[0])
                    
        #             if lateral >= self.lane.DEFAULT_WIDTH/2:
        #                 lateral = -self.lane.DEFAULT_WIDTH/2 # von lokalen lateralen KO der eigentlichen Spur zu denen der neuen Spur; ein bisschen unschoen aber sollte klappen
        #                 cur_lane_idx = goal_lane
        #             theta_lane = self.road.network.get_lane(cur_lane_idx).heading_at(longitudinal)
        #             # Integration der Geschwindigkeit zu globaler Position
        #             longitudinal += v_in_lane_direction*t
        #             lateral += v_orthogonal*t
        #             x, y = self.road.network.get_lane(cur_lane_idx).position(longitudinal, lateral)
        #             # Trajektorie
        #             trajectory.append((np.array([x,y]), theta_lane))
        #             # Index erhoehen
        #             idx += 1
        #             if np.linalg.norm(self.road.network.get_lane(goal_lane).position(longitudinal, 0)[1] - self.road.network.get_lane(cur_lane_idx).position(longitudinal, lateral)[1]) <= 0.3:
        #                 # wenn Abstand zwischen Mitte von Ziel-Lane und aktueller Position kleiner als 0.3m ist
        #                 break
        #         if idx <= len(times)-1: # prueft ob Praediktionshorizont schon erreicht wurde
        #             for t in times[idx:]:
        #                 longitudinal += self.speed*t # verwenden von skalarer Geschwindigkeit, da jetzt Mitte der Lane gefolgt wird
        #                 x, y = self.road.network.get_lane(goal_lane).position(longitudinal, 0)
        #                 theta_lane = self.road.network.get_lane(goal_lane).heading_at(longitudinal)
        #                 trajectory.append((np.array([x,y]), theta_lane))

        # else: 
        #     """
        #     Falls kein Spurwechsel praediziert, dann auf Spur bleiben oder Route folgen.
        #     Der Methode "position_heading_along_route" wird als lateraler Parameter 0 uebergeben -> geht von Fahren in Spurmitte aus
        #     """
        #     longitudinal = coordinates[0]
        #     for t in times:
        #         longitudinal += self.speed * t
        #         trajectory.append(self.road.network.position_heading_along_route(route, longitudinal, 0))

        # return tuple(zip(*trajectory))
        return (positions, headings)

    @property
    def v_direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading + self.beta), np.sin(self.heading + self.beta)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.v_direction # TODO: slip angle beta should be used here: UPDATE beta verwendet (15.08.2022)

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang])
        else:
            return np.zeros((3,))

    @property
    def target_lane_offset(self) -> np.ndarray:
        """
        eigene Methode um Abstand zur Soll-Lane zu ermitteln
        """
        target_lane_id = len(self.road.network.all_side_lanes(self.lane_index))-1 # groesster Spurindex = rechte Spur
        lane_index_id = self.lane_index[2]
        if self.lane is not None:
            if lane_index_id != target_lane_id:
                # wenn nicht auf Target-Lane
                # lokale Lane Koordinaten
                long, lat = self.lane.local_coordinates(self.position)
                # Anzahl Spuren bis Target-Lane
                n_lanes_to_target = target_lane_id - lane_index_id
                # lateraler Versatz bis Target-Lane (absolut)
                lat -= n_lanes_to_target*self.lane.DEFAULT_WIDTH  # immer negativer Abstand da ego-vehicle links von rechter Spur (Definition ist an Definition von local_coordinates() angelehnt) 
                ang = self.lane.local_angle(self.heading, long)  
                return np.array([long, lat, ang])
            else:
                # wenn schon auf Target-Lane dann selbe Methode wie lane_offset()
                long, lat = self.lane.local_coordinates(self.position)
                ang = self.lane.local_angle(self.heading, long)  
                return np.array([long, lat, ang]) 
        else:
            return np.zeros((3,))

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading, # Gierwinkel
            'cos_h': self.direction[0], # tatsaechliche x-Ausrichtung des vehicles berechnet durch Gierwinkel (nicht Ausrichtung der Geschwindigkeit)
            'sin_h': self.direction[1], # tatsaechliche y-Ausrichtung des vehicles berechnet durch Gierwinkel (nicht Ausrichtung der Geschwindigkeit)
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1],
            'long_off': self.lane_offset[0], # :return: int -> longitudonaler offset zwischen aktueller Position und Beginn der Lane in Richtung Ende der Lane
            'lat_off': self.lane_offset[1], # :return: int -> lateraler offset zwischen aktueller Position und Lane (Definiert als Mitte der Fahrbahn)
            'ang_off': self.lane_offset[2],
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()
