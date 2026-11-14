"""
Smart Workstation - Raspberry Pi Hardware Interface Layer
Course: 2110356 Embedded System (CEDT), Chulalongkorn University

Hardware Specifications:
- Sensors (Mandatory Requirement M1):
    1. HC-SR04 Ultrasonic Distance Sensor (Face-to-workstation posture measurement)
    2. BH1750 Digital Ambient Light Sensor (Reading surface illumination via I2C)
    3. KW11-3Z Limit Switch / Emergency Stop (Curtain physical boundary safety)
- Actuators (Mandatory Requirement M2):
    1. 28BYJ-48 Stepper Motor + ULN2003 (Motorized Blocker Curtain, half-step drive)
    2. 5V High-CRI LED Bar via N-Channel Logic MOSFET (Closed-loop PWM lighting)
    3. SG90 Micro Servo (Light Bar Tilt Angle Modulation)

Features:
- Dual-mode operation: Real Raspberry Pi GPIO / I2C or Mock Simulation mode.
- Non-blocking safety timeouts and coil de-energizing on stepper idle.
- Median filter on ultrasonic distance to reject acoustic multipath reflections.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Hardware] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HardwareInterface")

# Check for RPi.GPIO and smbus2 availability
_CAN_IMPORT_RPI = False
_CAN_IMPORT_SMBUS = False

try:
    import RPi.GPIO as GPIO  # type: ignore
    _CAN_IMPORT_RPI = True
except (ImportError, RuntimeError):
    GPIO = None

try:
    from smbus2 import SMBus  # type: ignore
    _CAN_IMPORT_SMBUS = True
except ImportError:
    SMBus = None


@dataclass
class PinConfig:
    """BCM Pin Numbering Configuration for Raspberry Pi 4 Model B."""
    # Ultrasonic HC-SR04
    ULTRASONIC_TRIG: int = 23
    ULTRASONIC_ECHO: int = 24  # 5V echo line stepped down to 3.3V via 1k/2k divider

    # Limit Switch / Emergency Button (KW11-3Z)
    LIMIT_SWITCH: int = 17     # Internal pull-up enabled (active LOW when closed)

    # 28BYJ-48 Stepper Motor via ULN2003
    STEPPER_IN1: int = 5
    STEPPER_IN2: int = 6
    STEPPER_IN3: int = 13
    STEPPER_IN4: int = 19

    # LED Light Bar (MOSFET Gate PWM)
    LED_PWM_PIN: int = 18      # Hardware PWM0 on RPi

    # SG90 Micro Servo (PWM)
    SERVO_PIN: int = 12        # Hardware PWM0 alternate on RPi

    # I2C Configuration for BH1750
    I2C_BUS: int = 1
    BH1750_I2C_ADDR: int = 0x23  # 0x23 (ADDR grounded) or 0x5C (ADDR connected to VCC)


class CurtainState:
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    OPENING = "OPENING"
    CLOSING = "CLOSING"
    UNKNOWN = "UNKNOWN"


class HardwareInterface:
    """
    Unified hardware controller for Raspberry Pi 4 Gateway.
    Handles sensor readings, motor stepping, PWM brightness tuning,
    servo tilting, and hardware interrupt-driven emergency stops.
    """

    # Half-step sequence for 28BYJ-48 stepper motor (8 phases)
    STEP_SEQUENCE = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]

    # Full travel for curtain (approx. 2.5 full rotations = 2.5 * 4096 = 10,240 half-steps)
    MAX_CURTAIN_STEPS = 10240
    HOMING_TIMEOUT_STEPS = 12000

    def __init__(self, pin_config: Optional[PinConfig] = None, mock: Optional[bool] = None) -> None:
        self.pins = pin_config or PinConfig()

        # Decide whether to run in mock mode
        env_mock = os.getenv("MOCK_HARDWARE", "").lower() in ("1", "true", "yes")
        if mock is not None:
            self.mock_mode = mock
        else:
            self.mock_mode = env_mock or not (_CAN_IMPORT_RPI and _CAN_IMPORT_SMBUS)

        # Thread synchronization
        self._lock = threading.Lock()
        self._motor_lock = threading.Lock()

        # Actuator states
        self.curtain_state = CurtainState.UNKNOWN
        self.curtain_position_steps = 0  # 0 = Fully OPEN, MAX_CURTAIN_STEPS = Fully CLOSED
        self.led_duty_cycle = 0.0        # 0.0% to 100.0%
        self.servo_angle = 90.0          # 0.0° to 180.0° (90° = neutral centered)
        self.is_emergency_stopped = False

        # Physical hardware handles
        self._led_pwm = None
        self._servo_pwm = None
        self._i2c_bus = None

        # Mock simulation state variables
        self._mock_distance = 45.0
        self._mock_base_lux = 250.0
        self._mock_limit_switch_pressed = False

        if self.mock_mode:
            logger.info(">>> Initialized in SIMULATION / MOCK HARDWARE MODE <<<")
            self.curtain_state = CurtainState.OPEN
            self.curtain_position_steps = 0
        else:
            logger.info("Initializing Physical Raspberry Pi GPIO and I2C interfaces...")
            self._init_physical_hardware()

    # =========================================================================
    # Physical Hardware Initialization
    # =========================================================================
    def _init_physical_hardware(self) -> None:
        """Configures BCM GPIO pin modes, PWM channels, and I2C bus."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # 1. Ultrasonic HC-SR04
        GPIO.setup(self.pins.ULTRASONIC_TRIG, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.pins.ULTRASONIC_ECHO, GPIO.IN)

        # 2. Limit Switch (KW11-3Z) with pull-up resistor and interrupt callback
        GPIO.setup(self.pins.LIMIT_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(
            self.pins.LIMIT_SWITCH,
            GPIO.FALLING,
            callback=self._on_limit_switch_triggered,
            bouncetime=200,
        )

        # 3. Stepper Motor Pins
        for pin in (
            self.pins.STEPPER_IN1,
            self.pins.STEPPER_IN2,
            self.pins.STEPPER_IN3,
            self.pins.STEPPER_IN4,
        ):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # 4. LED Bar PWM (1000 Hz for flicker-free illumination)
        GPIO.setup(self.pins.LED_PWM_PIN, GPIO.OUT)
        self._led_pwm = GPIO.PWM(self.pins.LED_PWM_PIN, 1000)
        self._led_pwm.start(0.0)

        # 5. SG90 Servo PWM (50 Hz standard hobby servo cycle)
        GPIO.setup(self.pins.SERVO_PIN, GPIO.OUT)
        self._servo_pwm = GPIO.PWM(self.pins.SERVO_PIN, 50)
        self._servo_pwm.start(7.5)  # 7.5% duty = ~90 deg neutral
        time.sleep(0.2)
        self._servo_pwm.ChangeDutyCycle(0)  # Stop pulse to avoid motor jitter

        # 6. BH1750 I2C
        try:
            self._i2c_bus = SMBus(self.pins.I2C_BUS)
            # Power ON BH1750 (Opcode 0x01)
            self._i2c_bus.write_byte(self.pins.BH1750_I2C_ADDR, 0x01)
            # Continuous H-Resolution Mode (Opcode 0x10, 1 lx resolution, 120ms measurement)
            self._i2c_bus.write_byte(self.pins.BH1750_I2C_ADDR, 0x10)
            time.sleep(0.15)
            logger.info("BH1750 Ambient Light Sensor online on I2C-1 addr 0x%02X", self.pins.BH1750_I2C_ADDR)
        except Exception as e:
            logger.error("Failed to initialize BH1750 on I2C bus: %s", e)

        logger.info("Physical hardware initialization complete.")

    def _on_limit_switch_triggered(self, channel: int) -> None:
        """Interrupt callback when curtain reaches physical limit switch."""
        logger.warning("SAFETY ALERT: Limit Switch (KW11-3Z) triggered on GPIO %d!", channel)
        self.is_emergency_stopped = True
        self._deenergize_stepper()

    # =========================================================================
    # Sensors (Requirement M1)
    # =========================================================================
    def read_distance_cm(self, samples: int = 3) -> float:
        """
        Reads face/head distance from HC-SR04 ultrasonic sensor.
        Applies a multi-sample median filter to eliminate acoustic reflection spikes.
        Returns: Distance in centimeters (2.0 to 300.0 cm).
        """
        if self.mock_mode:
            # Generate realistic slight jitter around current mock distance
            jitter = random.uniform(-0.8, 0.8)
            return round(max(5.0, min(200.0, self._mock_distance + jitter)), 1)

        readings = []
        for _ in range(samples):
            dist = self._single_ultrasonic_ping()
            if dist is not None and 2.0 <= dist <= 300.0:
                readings.append(dist)
            time.sleep(0.02)  # 20ms pause between pings to prevent reverberation

        if not readings:
            # Fallback if echo was missed
            return 999.0

        readings.sort()
        median = readings[len(readings) // 2]
        return round(median, 1)

    def _single_ultrasonic_ping(self) -> Optional[float]:
        """Sends a 10µs trigger pulse and measures echo pulse width with microsecond timeout."""
        with self._lock:
            # Send 10µs HIGH pulse to TRIG
            GPIO.output(self.pins.ULTRASONIC_TRIG, GPIO.LOW)
            time.sleep(0.000002)
            GPIO.output(self.pins.ULTRASONIC_TRIG, GPIO.HIGH)
            time.sleep(0.000010)
            GPIO.output(self.pins.ULTRASONIC_TRIG, GPIO.LOW)

            # Wait for ECHO to go HIGH (with 25ms timeout = ~4 meters)
            start_time = time.time()
            timeout_epoch = start_time + 0.025
            pulse_start = start_time

            while GPIO.input(self.pins.ULTRASONIC_ECHO) == 0:
                pulse_start = time.time()
                if pulse_start > timeout_epoch:
                    return None

            # Wait for ECHO to go LOW
            pulse_end = pulse_start
            timeout_epoch = pulse_start + 0.025
            while GPIO.input(self.pins.ULTRASONIC_ECHO) == 1:
                pulse_end = time.time()
                if pulse_end > timeout_epoch:
                    return None

            pulse_duration = pulse_end - pulse_start
            # Speed of sound at 20°C is ~343 m/s = 34300 cm/s
            distance = (pulse_duration * 34300.0) / 2.0
            return distance

    def read_lux(self) -> float:
        """
        Reads ambient reading surface illumination in Lux from BH1750.
        Returns: Light intensity in Lux (0.0 to 65535.0 Lux).
        """
        if self.mock_mode:
            # Ambient light increases proportionally with LED Bar duty cycle
            boost = (self.led_duty_cycle / 100.0) * 350.0
            noise = random.uniform(-4.0, 4.0)
            return round(max(0.0, self._mock_base_lux + boost + noise), 1)

        if self._i2c_bus is None:
            return 0.0

        try:
            with self._lock:
                data = self._i2c_bus.read_i2c_block_data(self.pins.BH1750_I2C_ADDR, 0x10, 2)
            raw_val = (data[0] << 8) | data[1]
            # BH1750 conversion factor: raw / 1.2
            lux = raw_val / 1.2
            return round(lux, 1)
        except Exception as e:
            logger.warning("Error reading BH1750: %s", e)
            return 0.0

    def is_limit_switch_pressed(self) -> bool:
        """Returns True if the physical curtain limit switch is triggered/pressed."""
        if self.mock_mode:
            return self._mock_limit_switch_pressed or (self.curtain_position_steps >= self.MAX_CURTAIN_STEPS)

        # Pin is pulled HIGH; pressing connects to GND (LOW)
        return GPIO.input(self.pins.LIMIT_SWITCH) == GPIO.LOW

    # =========================================================================
    # Actuators (Requirement M2)
    # =========================================================================
    def set_led_brightness(self, duty_percent: float) -> None:
        """
        Modulates the 5V LED Bar using PWM (0.0% to 100.0%).
        """
        duty_percent = max(0.0, min(100.0, float(duty_percent)))
        self.led_duty_cycle = round(duty_percent, 1)

        if not self.mock_mode and self._led_pwm is not None:
            self._led_pwm.ChangeDutyCycle(self.led_duty_cycle)

    def tune_lighting_closed_loop(self, target_lux: float = 450.0, tolerance: float = 25.0) -> float:
        """
        Proportional closed-loop controller maintaining ~450 Lux on study surface.
        Returns: Current measured lux after adjustment.
        """
        current_lux = self.read_lux()
        error = target_lux - current_lux

        if abs(error) > tolerance:
            # Proportional gain: adjust duty cycle based on illumination deficit
            kp = 0.15
            adjustment = error * kp
            new_duty = max(0.0, min(100.0, self.led_duty_cycle + adjustment))
            self.set_led_brightness(new_duty)

        return current_lux

    def set_servo_angle(self, angle_deg: float) -> None:
        """
        Sets the SG90 Micro Servo angle (0.0° to 180.0°) to tilt the light bar.
        0° = tilted downward toward desk, 90° = neutral, 180° = tilted upward.
        """
        angle_deg = max(0.0, min(180.0, float(angle_deg)))
        self.servo_angle = round(angle_deg, 1)

        if not self.mock_mode and self._servo_pwm is not None:
            # Map 0° -> 2.5% duty cycle, 180° -> 12.5% duty cycle
            duty = 2.5 + (self.servo_angle / 18.0)
            self._servo_pwm.ChangeDutyCycle(duty)
            time.sleep(0.15)
            # Sleep pulse to eliminate SG90 acoustic servo hum / vibration
            self._servo_pwm.ChangeDutyCycle(0)

    def step_curtain_motor(self, steps: int, direction_close: bool = True, step_delay: float = 0.0012) -> int:
        """
        Steps the 28BYJ-48 motor using half-stepping (8-phase) sequence.
        direction_close: True = lowering curtain (CLOSE), False = raising curtain (OPEN).
        Includes limit-switch safety break and automatic coil de-energizing.
        Returns: Actual steps moved.
        """
        with self._motor_lock:
            steps_moved = 0
            seq_len = len(self.STEP_SEQUENCE)
            step_order = range(seq_len) if direction_close else range(seq_len - 1, -1, -1)

            if direction_close:
                self.curtain_state = CurtainState.CLOSING
            else:
                self.curtain_state = CurtainState.OPENING

            try:
                for _ in range(steps):
                    # Check limit switch if closing
                    if direction_close and self.is_limit_switch_pressed():
                        logger.info("Curtain reached limit switch. Stopping motor.")
                        self.curtain_position_steps = self.MAX_CURTAIN_STEPS
                        self.curtain_state = CurtainState.CLOSED
                        break

                    if not direction_close and self.curtain_position_steps <= 0:
                        self.curtain_position_steps = 0
                        self.curtain_state = CurtainState.OPEN
                        break

                    for phase in step_order:
                        if not self.mock_mode:
                            pattern = self.STEP_SEQUENCE[phase]
                            GPIO.output(self.pins.STEPPER_IN1, pattern[0])
                            GPIO.output(self.pins.STEPPER_IN2, pattern[1])
                            GPIO.output(self.pins.STEPPER_IN3, pattern[2])
                            GPIO.output(self.pins.STEPPER_IN4, pattern[3])
                        time.sleep(step_delay)

                    steps_moved += 1
                    if direction_close:
                        self.curtain_position_steps += 1
                    else:
                        self.curtain_position_steps -= 1

            finally:
                # De-energize coils to prevent ULN2003 / 28BYJ-48 heat buildup
                self._deenergize_stepper()

            if self.curtain_position_steps >= self.MAX_CURTAIN_STEPS:
                self.curtain_state = CurtainState.CLOSED
            elif self.curtain_position_steps <= 0:
                self.curtain_state = CurtainState.OPEN

            return steps_moved

    def open_curtain(self) -> None:
        """Fully rolls up the blocker curtain (OPEN)."""
        if self.curtain_state == CurtainState.OPEN and self.curtain_position_steps == 0:
            return

        logger.info("Action: Rolling UP Blocker Curtain (OPEN)...")
        if self.mock_mode:
            self.curtain_position_steps = 0
            self.curtain_state = CurtainState.OPEN
            logger.info("[MOCK] Curtain is now fully OPEN.")
            return

        steps_needed = self.curtain_position_steps if self.curtain_position_steps > 0 else self.MAX_CURTAIN_STEPS
        self.step_curtain_motor(steps=steps_needed, direction_close=False)
        self.curtain_position_steps = 0
        self.curtain_state = CurtainState.OPEN
        logger.info("Curtain OPEN complete.")

    def close_curtain(self) -> None:
        """Lowers the blocker curtain to block improper posture/reading (CLOSE)."""
        if self.curtain_state == CurtainState.CLOSED and self.curtain_position_steps >= self.MAX_CURTAIN_STEPS:
            return

        logger.info("Action: Lowering Blocker Curtain (CLOSE)...")
        if self.mock_mode:
            self.curtain_position_steps = self.MAX_CURTAIN_STEPS
            self.curtain_state = CurtainState.CLOSED
            logger.info("[MOCK] Curtain is now fully CLOSED.")
            return

        steps_needed = self.MAX_CURTAIN_STEPS - self.curtain_position_steps
        self.step_curtain_motor(steps=max(100, steps_needed), direction_close=True)
        self.curtain_position_steps = self.MAX_CURTAIN_STEPS
        self.curtain_state = CurtainState.CLOSED
        logger.info("Curtain CLOSE complete.")

    def _deenergize_stepper(self) -> None:
        """De-energizes all four coils on the ULN2003 driver."""
        if not self.mock_mode:
            GPIO.output(self.pins.STEPPER_IN1, GPIO.LOW)
            GPIO.output(self.pins.STEPPER_IN2, GPIO.LOW)
            GPIO.output(self.pins.STEPPER_IN3, GPIO.LOW)
            GPIO.output(self.pins.STEPPER_IN4, GPIO.LOW)

    # =========================================================================
    # Mock Helpers (For unit tests and offline testing)
    # =========================================================================
    def set_mock_distance(self, dist_cm: float) -> None:
        self._mock_distance = float(dist_cm)

    def set_mock_base_lux(self, lux: float) -> None:
        self._mock_base_lux = float(lux)

    def set_mock_limit_switch(self, pressed: bool) -> None:
        self._mock_limit_switch_pressed = bool(pressed)

    # =========================================================================
    # Cleanup & Teardown
    # =========================================================================
    def cleanup(self) -> None:
        """Releases GPIO pins, stops PWM timers, and clears bus handles."""
        logger.info("Cleaning up hardware interfaces...")
        try:
            self.set_led_brightness(0.0)
            if self._led_pwm:
                self._led_pwm.stop()
            if self._servo_pwm:
                self._servo_pwm.stop()
            self._deenergize_stepper()

            if not self.mock_mode and GPIO:
                GPIO.cleanup()
            if not self.mock_mode and self._i2c_bus:
                self._i2c_bus.close()
        except Exception as e:
            logger.warning("Error during hardware cleanup: %s", e)
        logger.info("Hardware cleanup finished.")

