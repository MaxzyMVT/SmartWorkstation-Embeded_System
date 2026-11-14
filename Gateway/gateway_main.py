"""
Smart Workstation - Central Gateway Firmware & AI Control Daemon
Course: 2110356 Embedded System (CEDT), Chulalongkorn University

Runs exclusively on the Raspberry Pi 4 Gateway:
- Ingests wireless sensor packets from ESP32 Sensor Nodes via MQTT.
- Reads directly-wired sensors (HC-SR04 ultrasonic, BH1750 I2C lux, KW11-3Z limit switch).
- Executes on-device Machine Learning model (<10ms inference) every 1 second.
- Drives physical actuators in a closed loop (28BYJ-48 stepper curtain, LED PWM, SG90 servo).
- Publishes unified telemetry to `smart_workstation/telemetry` for the Web Dashboard.
- Dispatches alert notifications to Discord & LINE Webhooks on posture/environment violations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# Local modules
from hardware_interface import CurtainState, HardwareInterface
from train_ai_model import SmartWorkstationAI

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Gateway] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GatewayMain")

# Optional MQTT import with fallback
try:
    import paho.mqtt.client as mqtt  # type: ignore
    _HAS_PAHO = True
except ImportError:
    mqtt = None
    _HAS_PAHO = False


@dataclass
class Node1Data:
    """Telemetry from ESP32 Sensor Node 1 (Chair Presence)."""
    presence: int = 1
    sitting_duration_min: float = 0.0
    last_updated: float = 0.0


@dataclass
class Node2Data:
    """Telemetry from ESP32 Sensor Node 2 (Ambient Environment)."""
    temp_c: float = 24.5
    sound_db: float = 35.0
    last_updated: float = 0.0


class WebhookNotifier:
    """
    Dispatches asynchronous alert notifications to Discord and LINE
    with rate-limiting / cooldown to prevent message flooding.
    """

    def __init__(
        self,
        discord_url: Optional[str] = None,
        line_token: Optional[str] = None,
        cooldown_sec: float = 45.0,
    ) -> None:
        self.discord_url = discord_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self.line_token = line_token or os.getenv("LINE_NOTIFY_TOKEN", "")
        self.cooldown_sec = cooldown_sec
        self._last_alert_time: Dict[str, float] = {}
        self._lock = threading.Lock()

    def dispatch_alert(self, alert_key: str, title: str, description: str, fields: Optional[Dict[str, str]] = None) -> None:
        """Sends alert if cooldown period has elapsed for the given alert_key."""
        now = time.time()
        with self._lock:
            last_time = self._last_alert_time.get(alert_key, 0.0)
            if now - last_time < self.cooldown_sec:
                return  # Skip due to cooldown
            self._last_alert_time[alert_key] = now

        # Run dispatch in a background thread to prevent blocking control loop
        thread = threading.Thread(
            target=self._send_payloads,
            args=(title, description, fields),
            daemon=True,
        )
        thread.start()

    def _send_payloads(self, title: str, description: str, fields: Optional[Dict[str, str]]) -> None:
        # 1. Discord Webhook
        if self.discord_url and self.discord_url.startswith("http"):
            try:
                embed_fields = []
                if fields:
                    for k, v in fields.items():
                        embed_fields.append({"name": k, "value": str(v), "inline": True})

                payload = {
                    "username": "Smart Workstation Bot",
                    "avatar_url": "https://img.icons8.com/fluency/96/desk.png",
                    "embeds": [
                        {
                            "title": f"⚠️ {title}",
                            "description": description,
                            "color": 0xE11D48,  # Red/Rose
                            "fields": embed_fields,
                            "footer": {"text": "Chula CEDT Embedded System Final Project 2026"},
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    ],
                }
                requests.post(self.discord_url, json=payload, timeout=4.0)
                logger.info("Dispatched Discord Webhook alert: '%s'", title)
            except Exception as e:
                logger.warning("Failed to dispatch Discord Webhook: %s", e)

        # 2. LINE Notify
        if self.line_token:
            try:
                msg = f"\n[Smart Workstation]\n⚠️ {title}\n{description}"
                if fields:
                    for k, v in fields.items():
                        msg += f"\n- {k}: {v}"
                headers = {"Authorization": f"Bearer {self.line_token}"}
                requests.post(
                    "https://notify-api.line.me/api/notify",
                    data={"message": msg},
                    headers=headers,
                    timeout=4.0,
                )
                logger.info("Dispatched LINE Notify alert: '%s'", title)
            except Exception as e:
                logger.warning("Failed to dispatch LINE Notify: %s", e)


class SmartWorkstationGateway:
    """
    Central daemon managing hardware, MQTT communication, AI inference,
    and closed-loop physical actuation.
    """

    MQTT_TOPIC_NODE1 = "smart_workstation/node1"
    MQTT_TOPIC_NODE2 = "smart_workstation/node2"
    MQTT_TOPIC_TELEMETRY = "smart_workstation/telemetry"
    MQTT_TOPIC_COMMANDS = "smart_workstation/commands"

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        mock_hardware: bool = False,
        model_path: str = "model.joblib",
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.running = False

        # Subsystems
        self.hw = HardwareInterface(mock=mock_hardware)
        self.ai = SmartWorkstationAI(model_path=model_path)
        self.notifier = WebhookNotifier()

        # Thread-safe remote sensor telemetry
        self.node1 = Node1Data()
        self.node2 = Node2Data()
        self.state_lock = threading.Lock()

        # Tracking state
        self.consecutive_class_counts: Dict[int, int] = {i: 0 for i in range(7)}
        self.current_ai_state: Dict[str, Any] = {}
        self.last_curtain_action_time = 0.0

        # MQTT Client
        self.mqtt_client: Optional[Any] = None
        if _HAS_PAHO:
            self._init_mqtt()
        else:
            logger.warning("paho-mqtt not available. Running MQTT in mock/offline mode.")

    def _init_mqtt(self) -> None:
        client_id = f"rpi_gateway_{os.getpid()}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv311)
        else:
            self.mqtt_client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

    def _on_mqtt_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        # rc can be an integer or a ReasonCode object in paho-mqtt v2
        rc_val = getattr(rc, "value", rc)
        if rc_val == 0:
            logger.info("Connected to MQTT Broker at %s:%d", self.broker_host, self.broker_port)
            client.subscribe([(self.MQTT_TOPIC_NODE1, 0), (self.MQTT_TOPIC_NODE2, 0), (self.MQTT_TOPIC_COMMANDS, 0)])
        else:
            logger.error("Failed to connect to MQTT Broker, return code: %s", rc)

    def _on_mqtt_disconnect(self, client: Any, userdata: Any, *args: Any) -> None:
        logger.warning("Disconnected from MQTT Broker. Reconnecting...")

    def _on_mqtt_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """Parses incoming JSON payloads from Node 1, Node 2, or dashboard commands."""
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8", errors="ignore")

        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received on %s: %s", topic, payload_str)
            return

        now = time.time()
        with self.state_lock:
            if topic == self.MQTT_TOPIC_NODE1:
                # Sensor Node 1: Chair presence & session duration
                self.node1.presence = int(data.get("presence", data.get("seat", 1)))
                self.node1.sitting_duration_min = float(data.get("sitting_duration_min", data.get("duration", 0.0)))
                self.node1.last_updated = now

            elif topic == self.MQTT_TOPIC_NODE2:
                # Sensor Node 2: Ambient desk temperature & noise
                self.node2.temp_c = float(data.get("temp_c", data.get("temp", 24.5)))
                self.node2.sound_db = float(data.get("sound_db", data.get("sound", 35.0)))
                self.node2.last_updated = now

            elif topic == self.MQTT_TOPIC_COMMANDS:
                self._handle_manual_command(data)

    def _handle_manual_command(self, cmd: Dict[str, Any]) -> None:
        """Handles manual control commands from the Web Dashboard."""
        logger.info("Received manual dashboard command: %s", cmd)
        action = cmd.get("action", "").upper()
        if action == "OPEN_CURTAIN":
            self.hw.open_curtain()
        elif action == "CLOSE_CURTAIN":
            self.hw.close_curtain()
        elif action == "SET_LED":
            duty = float(cmd.get("duty", 50.0))
            self.hw.set_led_brightness(duty)
        elif action == "SET_SERVO":
            angle = float(cmd.get("angle", 90.0))
            self.hw.set_servo_angle(angle)

    def start(self) -> None:
        """Starts gateway background threads and enters the main 1Hz control loop."""
        self.running = True
        logger.info("=" * 65)
        logger.info("Starting Smart Workstation Gateway Daemon...")
        logger.info("Requirement M1: 2+ Local sensors (HC-SR04 ultrasonic + BH1750 lux + Limit switch)")
        logger.info("Requirement M2: Closed-loop actuators (28BYJ-48 curtain + LED bar + SG90 servo)")
        logger.info("Requirement M4: On-device ML Random Forest inference pipeline")
        logger.info("=" * 65)

        # Connect MQTT in a non-blocking background thread
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.connect_async(self.broker_host, self.broker_port, keepalive=60)
                self.mqtt_client.loop_start()
            except Exception as e:
                logger.warning("Could not initiate connection to MQTT broker (%s:%d): %s", self.broker_host, self.broker_port, e)

        # Home curtain to initial OPEN position
        self.hw.open_curtain()
        self.hw.set_servo_angle(85.0)

        try:
            while self.running:
                loop_start = time.time()
                self._control_iteration()
                elapsed = time.time() - loop_start
                sleep_time = max(0.05, 1.0 - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt.")
        finally:
            self.stop()

    def _control_iteration(self) -> None:
        """Single closed-loop iteration executed every 1 second."""
        # 1. Read local physical sensors (M1)
        distance_cm = self.hw.read_distance_cm()
        lux = self.hw.read_lux()

        # 2. Retrieve thread-safe wireless node data
        with self.state_lock:
            # Check staleness (if no packet received in 15 seconds, use fallback/simulation)
            now = time.time()
            if self.hw.mock_mode and (now - self.node1.last_updated > 10.0):
                # Update mock sitting duration
                self.node1.sitting_duration_min = round(self.node1.sitting_duration_min + (1.0 / 60.0), 2)
            presence = self.node1.presence
            duration_min = self.node1.sitting_duration_min
            temp_c = self.node2.temp_c
            sound_db = self.node2.sound_db

        # 3. AI Prediction Pipeline (M4)
        ai_res = self.ai.predict(
            presence=presence,
            distance_cm=distance_cm,
            lux=lux,
            temp_c=temp_c,
            sound_db=sound_db,
            sitting_duration_min=duration_min,
        )
        self.current_ai_state = ai_res
        class_id = ai_res["class_id"]
        class_name = ai_res["class_name"]

        # Track consecutive detections to filter transient blips
        for c in range(7):
            if c == class_id:
                self.consecutive_class_counts[c] += 1
            else:
                self.consecutive_class_counts[c] = 0

        # 4. Closed-Loop Actuation (M2)
        self._execute_actuation_logic(class_id, distance_cm, lux, temp_c, sound_db, duration_min)

        # 5. Publish unified telemetry for Web Dashboard & Node-RED
        self._publish_telemetry(presence, distance_cm, lux, temp_c, sound_db, duration_min, ai_res)

    def _execute_actuation_logic(
        self,
        class_id: int,
        distance_cm: float,
        lux: float,
        temp_c: float,
        sound_db: float,
        duration_min: float,
    ) -> None:
        """Drives stepper curtain, PWM lighting, and servo based on classified state."""

        # Class 0: OPTIMAL_STUDY
        if class_id == 0:
            if self.hw.curtain_state != CurtainState.OPEN:
                self.hw.open_curtain()
            # Tune illumination in closed-loop to reach ~450 Lux
            self.hw.tune_lighting_closed_loop(target_lux=450.0)
            self.hw.set_servo_angle(85.0)

        # Class 1: TEXT_NECK_TOO_CLOSE (Slouching forward < 28cm)
        elif class_id == 1:
            if self.consecutive_class_counts[1] >= 2:  # 2 consecutive seconds confirms text neck
                if self.hw.curtain_state != CurtainState.CLOSED:
                    logger.warning("AI ACTION: Closing Curtain due to TEXT_NECK_TOO_CLOSE (%.1f cm)", distance_cm)
                    self.hw.close_curtain()
                    self.notifier.dispatch_alert(
                        alert_key="TEXT_NECK",
                        title="Text-Neck Posture Warning",
                        description="Your head is too close to the reading surface (<28 cm). The blocker curtain has dropped.",
                        fields={"Distance": f"{distance_cm:.1f} cm", "Threshold": "< 28 cm", "Action": "Curtain Closed"},
                    )

        # Class 2: SLOUCH_TOO_FAR (Leaning back > 65cm)
        elif class_id == 2:
            if self.consecutive_class_counts[2] >= 2:
                if self.hw.curtain_state != CurtainState.CLOSED:
                    logger.warning("AI ACTION: Closing Curtain due to SLOUCH_TOO_FAR (%.1f cm)", distance_cm)
                    self.hw.close_curtain()
                    self.notifier.dispatch_alert(
                        alert_key="SLOUCH_FAR",
                        title="Slouching Warning",
                        description="You are leaning excessively far back (>65 cm). Please align your posture.",
                        fields={"Distance": f"{distance_cm:.1f} cm", "Threshold": "> 65 cm", "Action": "Curtain Closed"},
                    )

        # Class 3: POOR_LIGHTING (<200 Lux or >1200 Lux)
        elif class_id == 3:
            if lux < 200.0:
                # Attempt to boost LED Bar to 100%
                self.hw.set_led_brightness(100.0)
                if self.consecutive_class_counts[3] >= 3:
                    self.notifier.dispatch_alert(
                        alert_key="LOW_LIGHT",
                        title="Insufficient Lighting Alert",
                        description=f"Desk illumination is critically low ({lux:.0f} Lux). LED booster set to 100%.",
                        fields={"Current Lux": f"{lux:.0f} Lux", "Recommended": "400 - 600 Lux"},
                    )
            elif lux > 1200.0:
                # Glare protection: dim LED bar
                self.hw.set_led_brightness(0.0)
                if self.consecutive_class_counts[3] >= 3:
                    self.notifier.dispatch_alert(
                        alert_key="HIGH_GLARE",
                        title="Harsh Glare Warning",
                        description=f"Reading surface glare exceeds safe limits ({lux:.0f} Lux).",
                        fields={"Current Lux": f"{lux:.0f} Lux", "Action": "LED Dimmed"},
                    )

        # Class 4: NOISY_OR_HOT_FATIGUE (Sound > 60dB or Temp > 30°C)
        elif class_id == 4:
            if self.consecutive_class_counts[4] >= 3:
                self.notifier.dispatch_alert(
                    alert_key="ENV_FATIGUE",
                    title="Environmental Fatigue Warning",
                    description="Desk conditions are causing mental strain (High noise or excessive temperature).",
                    fields={"Temperature": f"{temp_c:.1f} °C", "Noise": f"{sound_db:.1f} dB"},
                )

        # Class 5: BREAK_TIMER_REACHED (>= 50 min continuous study)
        elif class_id == 5:
            if self.hw.curtain_state != CurtainState.CLOSED:
                logger.info("AI ACTION: 50-Minute Study Session Elapsed! Dropping curtain for 5-minute break.")
                self.hw.close_curtain()
                self.notifier.dispatch_alert(
                    alert_key="BREAK_TIME",
                    title="Study Break Interval Reached",
                    description="50 minutes of continuous study completed! Stand up, stretch, and give your eyes a 5-minute rest.",
                    fields={"Study Time": f"{duration_min:.1f} min", "Action": "Curtain Closed (Break Mode)"},
                )

        # Class 6: AWAY (User left desk)
        elif class_id == 6:
            # Turn off lighting to save power; park curtain open
            if self.hw.led_duty_cycle > 0:
                self.hw.set_led_brightness(0.0)
            if self.hw.curtain_state != CurtainState.OPEN:
                self.hw.open_curtain()

    def _publish_telemetry(
        self,
        presence: int,
        distance_cm: float,
        lux: float,
        temp_c: float,
        sound_db: float,
        duration_min: float,
        ai_res: Dict[str, Any],
    ) -> None:
        """Publishes consolidated telemetry formatted for React App.jsx."""
        payload = {
            # Direct mapping to Interface/src/App.jsx state variables
            "seat": bool(presence == 1),
            "distance": round(distance_cm, 1),
            "light": round(lux, 1),
            "temp": round(temp_c, 1),
            "sound": round(sound_db, 1),
            # Extended telemetry
            "sitting_duration_min": round(duration_min, 1),
            "curtain_state": self.hw.curtain_state,
            "led_duty_cycle": round(self.hw.led_duty_cycle, 1),
            "servo_angle": round(self.hw.servo_angle, 1),
            "ai_state": {
                "class_id": ai_res["class_id"],
                "class_name": ai_res["class_name"],
                "confidence": ai_res["confidence"],
                "action": ai_res["action"],
                "advice": ai_res["advice"],
                "latency_ms": ai_res["latency_ms"],
            },
            "timestamp": time.time(),
        }

        # Log iteration summary
        logger.info(
            "State: %-20s | Seat: %d | Dist: %5.1fcm | Lux: %5.1f | Temp: %4.1f°C | Sound: %4.1fdB | Curtain: %-7s | LED: %3.0f%%",
            ai_res["class_name"],
            presence,
            distance_cm,
            lux,
            temp_c,
            sound_db,
            self.hw.curtain_state,
            self.hw.led_duty_cycle,
        )

        if self.mqtt_client is not None and self.mqtt_client.is_connected():
            try:
                self.mqtt_client.publish(
                    self.MQTT_TOPIC_TELEMETRY,
                    json.dumps(payload),
                    qos=0,
                )
            except Exception as e:
                logger.warning("Failed to publish telemetry to MQTT: %s", e)

    def stop(self) -> None:
        """Gracefully shuts down network connections and physical actuators."""
        logger.info("Stopping Smart Workstation Gateway...")
        self.running = False
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
        self.hw.cleanup()
        logger.info("Gateway daemon stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Workstation Raspberry Pi Gateway & AI Daemon")
    parser.add_argument("--broker", type=str, default=os.getenv("MQTT_BROKER", "localhost"), help="MQTT broker hostname")
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)), help="MQTT broker port")
    parser.add_argument("--mock", action="store_true", help="Force hardware simulation / mock mode")
    parser.add_argument("--model", type=str, default=str(Path(__file__).parent / "model.joblib"), help="Path to trained AI model")
    args = parser.parse_args()

    gateway = SmartWorkstationGateway(
        broker_host=args.broker,
        broker_port=args.port,
        mock_hardware=args.mock,
        model_path=args.model,
    )

    # Attach signal handlers for graceful shutdown on SIGINT and SIGTERM
    def signal_handler(sig: int, frame: Any) -> None:
        logger.info("Caught signal %d. Triggering graceful shutdown...", sig)
        gateway.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    gateway.start()


if __name__ == "__main__":
    main()

