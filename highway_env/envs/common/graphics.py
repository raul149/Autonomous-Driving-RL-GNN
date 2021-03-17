import os
import time
from typing import TYPE_CHECKING, Callable, List
import numpy as np
import pygame
from gym.spaces import Discrete
import torch

from highway_env.envs.common.action import ActionType, DiscreteMetaAction, ContinuousAction
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.envs.common.observation import KinematicObservation
import models_eshagh
if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action

class EnvViewer(object):

    """A viewer to render a highway driving environment."""

    SAVE_IMAGES = False

    def __init__(self, env: 'AbstractEnv') -> None:
        self.env = env
        self.offscreen = env.config["offscreen_rendering"]

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.env.config["screen_width"], self.env.config["screen_height"])

        # A display is not mandatory to draw things. Ignoring the display.set_mode()
        # instruction allows the drawing to be done on surfaces without
        # handling a screen display, useful for e.g. cloud computing
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.env.config["screen_width"], self.env.config["screen_height"]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = env.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = env.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None
        self.start = False
        self.start2 = False
        self.q_network = None
        self.target_network = None


    def set_agent_display(self, agent_display: Callable) -> None:
        """
        Set a display callback provided by an agent

        So that they can render their behaviour on a dedicated agent surface, or even on the simulation surface.

        :param agent_display: a callback provided by the agent to display on surfaces
        """
        if self.agent_display is None:
            if not self.offscreen:
                if self.env.config["screen_width"] > self.env.config["screen_height"]:
                    self.screen = pygame.display.set_mode((self.env.config["screen_width"],
                                                           2 * self.env.config["screen_height"]))
                else:
                    self.screen = pygame.display.set_mode((2 * self.env.config["screen_width"],
                                                           self.env.config["screen_height"]))
            self.agent_surface = pygame.Surface((self.env.config["screen_width"], self.env.config["screen_height"]))
        self.agent_display = agent_display

    def set_agent_action_sequence(self, actions: List['Action']) -> None:
        """
        Set the sequence of actions chosen by the agent, so that it can be displayed

        :param actions: list of action, following the env's action space specification
        """
        if isinstance(self.env.action_space, Discrete):
            actions = [self.env.ACTIONS[a] for a in actions]
        if len(actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(actions,
                                                                          1 / self.env.config["policy_frequency"],
                                                                          1 / 3 / self.env.config["policy_frequency"],
                                                                          1 / self.env.config["simulation_frequency"])

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return
        if self.start:
            model_config = {
                "type": "EgoAttentionNetwork",
                "feature_size": 64,
                "embedding_layer": {
                    "type": "MultiLayerPerceptron",
                    "layers": [64, 64],
                    "reshape": False,
                    "in": 7
                },
                "others_embedding_layer": {
                    "type": "MultiLayerPerceptron",
                    "layers": [64, 64],
                    "reshape": False,
                    "in": 7
                },
                "self_attention_layer": {
                    "type": "SelfAttention",
                    "feature_size": 64,
                    "heads": 2
                },
                "attention_layer": {
                    "type": "EgoAttention",
                    "feature_size": 64,
                    "heads": 2
                },
                "output_layer": {
                    "type": "MultiLayerPerceptron",
                    "layers": [64, 64],
                    "reshape": False,
                },
                # "heads": 2,
                # "dropout_factor": 0
                "out": 3

            }
            self.q_network = models_eshagh.EgoAttentionNetwork(model_config).to('cpu')
            self.target_network = models_eshagh.EgoAttentionNetwork(model_config).to('cpu')
            self.target_network.load_state_dict(self.q_network.state_dict())
            torch.save(self.q_network.state_dict(), "/u/05/aznarr1/unix/Documents/Projectintersection/highway-env-master/model" + "/q_network_1_3_2.pt")
            self.start=False

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )
        """load_path = "/u/05/aznarr1/unix/Documents/Projectintersection/highway-env-master/model"
        self.q_network.load_state_dict(torch.load(load_path + "/q_network_1_3_2.pt"))


        observat = KinematicObservation(self.env).observe()
        logits, aaaamatrix, selfattmatrixs = self.q_network.forward(observat.reshape((1,)+observat.shape))
        action = torch.argmax(logits, dim=1).tolist()[0]
        #print(logits)
        #print('graph',aaaamatrix)
        aaaamatrix = torch.squeeze(aaaamatrix)

        aamatrix = aaaamatrix.cpu().detach().numpy()
        selfamatrix = torch.squeeze(selfattmatrixs)
        smatrix = selfamatrix.cpu().detach().numpy()
        #print(smatrix.shape)
        #print('FOR TEST',observat)
        #print(aamatrix.shape)
        #xxx = dqn_self_attention.attmatrix(observat)
        #print(xxx)
        """""" position = [*self.sim_surface.pos2pix(observat[1][1]*100, observat[1][2]*100)]
        if observat[2][0]==1:
            position2 = [*self.sim_surface.pos2pix(observat[2][1]*100, observat[2][2]*100)]
        else:
            if observat[2][0]==1:
                position2 = [*self.sim_surface.pos2pix(observat[2][1]*100, observat[2][2]*100)]
            else:
                if observat[3][0]==1:
                    position2 = [*self.sim_surface.pos2pix(observat[3][1]*100, observat[3][2]*100)]
                else:
                    if observat[4][0]==1:
                        position2 = [*self.sim_surface.pos2pix(observat[4][1]*100, observat[4][2]*100)]
                    else:
                        if observat[5][0]==1:
                            position2 = [*self.sim_surface.pos2pix(observat[5][1]*100, observat[5][2]*100)]
                        else:
                            position2 = [*self.sim_surface.pos2pix(observat[6][1]*100, observat[6][2]*100)]
        print(position)
        print(position2)
        #pygame.draw.rect(self.sim_surface, (50, 200, 0), position)
        pygame.draw.line(self.sim_surface, (50, 200, 0), position,position2, 10) """"""
        #print(x)
        #print()
        originx=observat[0][1]*100
        originy=observat[0][2]*100
        #origgiinn=PositionType([originx, originy])
        self.sim_surface.move_display_window_to2(originx,originy)
        i=0
        j=0
        print(aamatrix)
        print(smatrix)
        while i < 15:
            j=0
            while j < 15:
                if observat[i][0]>0 and observat[j][0]>0:
                    if i==j:
                        position = self.sim_surface.pos2pix(observat[i][1]*100, observat[i][2]*100)
                        position2 = self.sim_surface.pos2pix(observat[j][1]*100, observat[j][2]*100)
                        pygame.draw.circle(self.sim_surface, (50, 200, 0), position, int(smatrix[0][i][j]*20))
                        pygame.draw.circle(self.sim_surface, (50, 0, 200), position, int(smatrix[1][i][j]*20))
                    position = self.sim_surface.pos2pix(observat[i][1]*100, observat[i][2]*100)
                    position2 = self.sim_surface.pos2pix(observat[j][1]*100, observat[j][2]*100)
                    pygame.draw.line(self.sim_surface, (50, 200, 0), position, position2, int(smatrix[0][i][j]*10))
                    pygame.draw.line(self.sim_surface, (50, 00, 200), position, position2, int(smatrix[1][i][j]*10))
                j=j+1
            i=i+1
        k=0
        while k < 15:
            position = self.sim_surface.pos2pix(observat[0][1]*100, observat[0][2]*100)
            position2 = self.sim_surface.pos2pix(observat[k][1]*100, observat[k][2]*100)
            pygame.draw.line(self.sim_surface, (200, 200, 0), position, position2, int(smatrix[0][0][k]*10))
            pygame.draw.line(self.sim_surface, (200, 200, 200), position, position2, int(smatrix[1][0][k]*10))
            k=k+1

        position = self.sim_surface.pos2pix(observat[0][1]*100, observat[0][2]*100)
        pygame.draw.circle(self.sim_surface, (200, 200, 0), position, int(smatrix[0][0][0]*20))
        pygame.draw.circle(self.sim_surface, (200, 200, 200), position, int(smatrix[1][0][0]*20))

        k=0
        while k < 15:
            position = self.sim_surface.pos2pix(observat[0][1]*100, observat[0][2]*100)
            position2 = self.sim_surface.pos2pix(observat[k][1]*100, observat[k][2]*100)
            pygame.draw.line(self.sim_surface, (200, 0, 0), position, position2, int(aamatrix[0][k]*10))
            pygame.draw.line(self.sim_surface, (250, 150, 30), position, position2, int(aamatrix[1][k]*10))
            k=k+1

        position = self.sim_surface.pos2pix(observat[0][1]*100, observat[0][2]*100)
        pygame.draw.circle(self.sim_surface, (200, 0, 0), position, int(aamatrix[0][0]*20))
        pygame.draw.circle(self.sim_surface, (250, 150, 30), position, int(aamatrix[1][0]*20))"""


        """position3 = self.sim_surface.pos2pix(observat[2][1]*100, observat[2][2]*100)
                position4 = self.sim_surface.pos2pix(observat[3][1]*100, observat[3][2]*100)
                position5 = self.sim_surface.pos2pix(observat[4][1]*100, observat[4][2]*100)
                position6 = self.sim_surface.pos2pix(observat[5][1]*100, observat[5][2]*100)
                position7 = self.sim_surface.pos2pix(observat[6][1]*100, observat[6][2]*100)
                position8 = self.sim_surface.pos2pix(observat[7][1]*100, observat[7][2]*100)
                position9 = self.sim_surface.pos2pix(observat[8][1]*100, observat[8][2]*100)
                position10 = self.sim_surface.pos2pix(observat[9][1]*100, observat[9][2]*100)


                pygame.draw.line(self.sim_surface, (50, 200, 0), position, position3, int(aamatrix[0][2]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 0), position, position4, int(aamatrix[0][3]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 0), position, position5, int(aamatrix[0][4]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 0), position, position6, int(aamatrix[0][5]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 0), position, position7, int(aamatrix[0][6]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 0), position, position8, int(aamatrix[0][7]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 0), position, position9, int(aamatrix[0][8]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 200), position, position2, int(aamatrix[1][1]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 200), position, position3, int(aamatrix[1][2]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 200), position, position4, int(aamatrix[1][3]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 200), position, position5, int(aamatrix[1][4]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 200), position, position6, int(aamatrix[1][5]*10))
                pygame.draw.line(self.sim_surface, (50, 200, 200), position, position7, int(aamatrix[1][6]*10))"""
        self.sim_surface.move_display_window_to(self.window_position())
        #print(position)
        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.env.config["screen_width"] > self.env.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.env.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.env.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1
        #time.sleep(0.2)

    def get_image(self) -> np.ndarray:
        """the rendered image as a rbg array."""
        surface = self.screen if self.env.config["render_agent"] and not self.offscreen else self.sim_surface
        data = pygame.surfarray.array3d(surface)
        return np.moveaxis(data, 0, 1)

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()


class EventHandler(object):
    @classmethod
    def handle_event(cls, action_type: ActionType, event: pygame.event.EventType) -> None:
        """
        Map the pygame keyboard events to control decisions

        :param action_type: the ActionType that defines how the vehicle is controlled
        :param event: the pygame event
        """
        if isinstance(action_type, DiscreteMetaAction):
            cls.handle_discrete_action_event(action_type, event)
        elif isinstance(action_type, ContinuousAction):
            cls.handle_continuous_action_event(action_type, event)

    @classmethod
    def handle_discrete_action_event(cls, action_type: DiscreteMetaAction, event: pygame.event.EventType) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["FASTER"])
            if event.key == pygame.K_LEFT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["SLOWER"])
            if event.key == pygame.K_DOWN and action_type.lateral:
                action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            if event.key == pygame.K_UP:
                action_type.act(action_type.actions_indexes["LANE_LEFT"])

    @classmethod
    def handle_continuous_action_event(cls, action_type: ContinuousAction, event: pygame.event.EventType) -> None:
        action = action_type.last_action.copy()
        steering_index = action_type.space().shape[0] - 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0.7
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = -0.7
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = -0.7
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0.7
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = 0
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0
        action_type.act(action)
