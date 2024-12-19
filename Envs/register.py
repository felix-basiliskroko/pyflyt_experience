from gymnasium.envs.registration import register


# Register the environments
def register_custom_envs():
    register(
        id="Quadx-Waypoint-v0",
        entry_point="Envs.WaypointEnv:QuadXWaypoint",
        max_episode_steps=555,  # 2 * mu * 2sigma (gathered from 300 eval episodes)
    )

    register(
        id="SingleWaypointQuadXEnv-v0",
        entry_point="Envs.static_waypoint:SingleWaypointQuadXEnv",
        max_episode_steps=555,  # 2 * mu * 2sigma (gathered from 300 eval episodes)
    )


register_custom_envs()
