"""
Ray of HC-SR04 ultrasonic sensors.

Each sensor is treated as one directional ray.
The sensors can be placed around the robot at different angles.
"""

from dataclasses import dataclass
from time import sleep, time


@dataclass
class UltrasonicSensor:
    name: str
    trig_pin: int
    echo_pin: int
    angle_deg: float
    distance_cm: float = 999.0


# Put your real GPIO pin numbers here
# Example:
# - left sensor trig/echo pins
# - front sensor trig/echo pins
# - right sensor trig/echo pins
sensors = [
    UltrasonicSensor(
        name="left",
        trig_pin=0,      # TODO: put LEFT sensor TRIG pin here
        echo_pin=0,      # TODO: put LEFT sensor ECHO pin here
        angle_deg=-45.0
    ),
    UltrasonicSensor(
        name="front",
        trig_pin=0,      # TODO: put FRONT sensor TRIG pin here
        echo_pin=0,      # TODO: put FRONT sensor ECHO pin here
        angle_deg=0.0
    ),
    UltrasonicSensor(
        name="right",
        trig_pin=0,      # TODO: put RIGHT sensor TRIG pin here
        echo_pin=0,      # TODO: put RIGHT sensor ECHO pin here
        angle_deg=45.0
    ),
]


NO_DETECTION_DISTANCE_CM = 999.0
OBSTACLE_THRESHOLD_CM = 20.0


def read_distance_cm(sensor: UltrasonicSensor) -> float:
    """
    Reads distance from one HC-SR04 sensor.

    This function is prepared for Raspberry Pi GPIO logic.
    If you use Arduino/ESP32 instead, this file should not be Python.
    """

    # TODO:
    # Here we will later add real GPIO code:
    # 1. set TRIG low
    # 2. send 10 microsecond pulse on TRIG
    # 3. measure ECHO pulse duration
    # 4. convert duration to distance in cm

    # Temporary test value so the file works without hardware:
    return NO_DETECTION_DISTANCE_CM


def update_sensor_ray() -> None:
    """
    Updates distance value for every sensor in the ray.
    """

    for sensor in sensors:
        sensor.distance_cm = read_distance_cm(sensor)
        sleep(0.06)


def get_closest_sensor() -> UltrasonicSensor:
    """
    Returns the sensor that detected the closest object.
    """

    return min(sensors, key=lambda sensor: sensor.distance_cm)


def get_closest_distance() -> float:
    """
    Returns the closest detected distance.
    """

    closest_sensor = get_closest_sensor()
    return closest_sensor.distance_cm


def is_obstacle_detected() -> bool:
    """
    Checks if any sensor detected an obstacle closer than the threshold.
    """

    return get_closest_distance() < OBSTACLE_THRESHOLD_CM


def get_sensor_data() -> dict:
    """
    Returns all sensor readings as a dictionary.
    Useful later for sending data to the backend.
    """

    closest_sensor = get_closest_sensor()

    return {
        "sensors": [
            {
                "name": sensor.name,
                "angle_deg": sensor.angle_deg,
                "distance_cm": sensor.distance_cm,
            }
            for sensor in sensors
        ],
        "closest_sensor": closest_sensor.name,
        "closest_distance_cm": closest_sensor.distance_cm,
        "obstacle_detected": is_obstacle_detected(),
    }


def print_sensor_ray() -> None:
    """
    Prints current readings in the terminal.
    """

    print("=== Sensor ray readings ===")

    for sensor in sensors:
        print(
            f"{sensor.name} | "
            f"angle: {sensor.angle_deg} deg | "
            f"distance: {sensor.distance_cm} cm"
        )

    closest_sensor = get_closest_sensor()

    print(
        f"Closest obstacle: {closest_sensor.name} | "
        f"{closest_sensor.distance_cm} cm"
    )

    if is_obstacle_detected():
        print("Obstacle detected!")
    else:
        print("No obstacle nearby.")

    print()


if __name__ == "__main__":
    while True:
        update_sensor_ray()
        print_sensor_ray()
        sleep(0.2)