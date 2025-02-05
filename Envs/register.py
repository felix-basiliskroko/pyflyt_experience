from gymnasium.envs.registration import register


def register_custom_envs():
    register(
        id="SingleWaypointQuadXEnv-v0",
        entry_point="Envs.static_waypoint:SingleWaypointQuadXEnv",
    )


register_custom_envs()
