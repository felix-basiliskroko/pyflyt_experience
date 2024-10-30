from gymnasium.envs.registration import register


# Register the environments
def register_custom_envs():
    register(
        id="Quadx-Waypoint-v0",
        entry_point="Envs.WaypointEnv:QuadXWaypoint",
    )

    register(
        id="SingleWaypointQuadXEnv-v0",
        entry_point="Envs.static_waypoint:SingleWaypointQuadXEnv",
    )


register_custom_envs()
