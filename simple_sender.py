""" Script to send simulation instructions to Azure.

See simple_receiver for information.
"""

import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

# pip install azure-iot-device
from azure.iot.device import IoTHubDeviceClient, Message

# pip install azure-eventhub
from azure.eventhub import EventHubConsumerClient, EventData

# pip install azure-storage-blob
from azure.storage.blob import BlobClient


class InstructionSender:
    """Send information that can be interpreted as instructions for a simulation"""

    def __init__(self, client):
        self._client = client

    def measurements(self):
        # Call runscript for simulator here?
        pressure_values = [42, 1529, 1683]
        return {"pressure": pressure_values}

    def make_message(self, tp):
        return {
            "type": tp,
            "time": datetime.now(timezone.utc).isoformat(),
            "instruction": self.measurements(),
        }

    def send_data(self, tp="simulation_instruction"):
        data = self.make_message(tp)
        message = Message(json.dumps(data))
        self._client.send_message(message)
        print("Simulation data send at ", data["time"])


def main():
    # Store the passwords in a file config.json, with the format
    # {
    # "access_send" = password_send,
    # "access_receive" = password_receive
    # }

    with open("config.json", "r") as f:
        config = json.load(f)

    client_send = IoTHubDeviceClient.create_from_connection_string(
        config["access_send"]
    )
    sender = InstructionSender(client_send)

    sender.send_data("telemetry")
    sender.send_data()
    sender.send_data("telemetry")
    time.sleep(2)

    sender.send_data()


if __name__ == "__main__":
    main()
